import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import url_map, url_map_advprop, get_model_params
import numpy as np
from utils_HSI import Averagvalue
from torch import optim
from torch.distributions import Normal, Independent
class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)

    def forward(self, x):
        return self.attention(x)
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )   #
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[3] != skip.shape[3]:
                k = 0
                padding_size = skip.size(3) - x.size(3)
                if padding_size % 2 != 0:
                    padding_size += 1
                    k = -1
                padding_shape = (0, padding_size, 0, 0, 0, 0, 0, 0)
                x_padded = F.pad(x, padding_shape, 'constant', 0)
                if k == -1:
                    x = x_padded[:, :, :, :-1]
                else:
                    x = x_padded
            if x.shape[2] != skip.shape[2]:
                k = 0
                padding_size = x.size(2) - skip.size(2)
                if padding_size >= 0:
                    if padding_size % 2 != 0:
                        padding_size += 1
                        k = -1
                    padding_shape = (0, 0, 0, padding_size, 0, 0, 0, 0)
                    skip_padded = F.pad(skip, padding_shape, 'constant', 0)
                    if k == -1:
                        skip = skip_padded[:, :, :-1, :]
                    else:
                        skip = skip_padded
                else:
                    padding_size = -padding_size
                    if padding_size % 2 != 0:
                        padding_size += 1
                        k = -1
                    padding_shape = (0, 0, 0, padding_size, 0, 0, 0, 0)
                    x_padded = F.pad(x, padding_shape, 'constant', 0)
                    if k == -1:
                        x = x_padded[:, :, :-1, :]
                    else:
                        x = x_padded
            x = torch.cat([x, skip], dim=1)  # 上采样后的张量 x 会沿着通道维度（维度1）与 skip 进行拼接。
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetPlusPlusDecoder(nn.Module):   #unet++解码器模块
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,   #块数
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]

        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels
        self.center = nn.Identity()


        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(layer_idx + 1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]
                blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)

        blocks[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
            self.in_channels[-1], 0, self.out_channels[-1], **kwargs
        )
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1

    def forward(self, *features):
        features = features[1:]
        features = features[::-1]
        dense_x = {}

        for layer_idx in range(len(self.in_channels) - 1):
            for depth_idx in range(self.depth - layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f"x_{depth_idx}_{depth_idx}"](features[depth_idx], features[depth_idx + 1])
                    dense_x[f"x_{depth_idx}_{depth_idx}"] = output
                else:
                    new_list = []
                    for i in range(len(features)):
                        new_list.append(features[i])
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    z = 0
                    for u in range(len(cat_features)):
                        m = 0
                        if cat_features[u].size(2) != features[dense_l_i + 1].shape[2]:
                            if z == 0:
                                new_list = []
                            padding_size = cat_features[u].size(2) - features[dense_l_i + 1].size(2)
                            if padding_size % 2 != 0:
                                padding_size += 1
                                m = -1
                            padding_shape = (0, 0, 0, padding_size, 0, 0, 0, 0)
                            expend = F.pad(features[dense_l_i + 1 + u], padding_shape, 'constant', 0)
                            if u == 0:
                                for i in range(len(features[:dense_l_i + 1])):
                                    new_list.append(features[:dense_l_i + 1][i])
                                if m == -1:
                                    new_list.append(expend[:, :, :-1, :])
                                else:
                                    new_list.append(expend)
                                for i in range(len(features[dense_l_i + 2:])):
                                    new_list.append(features[dense_l_i + 2:][i])
                                z = 1
                                en_list = [new_list[dense_l_i + 1]]
                            else:
                                if m == -1:
                                    new_list[dense_l_i + 1 + u] = expend[:, :, :-1, :]
                                else:
                                    new_list[dense_l_i + 1 + u] = expend
                                en_list.append(new_list[dense_l_i + 1])
                        else:
                            if u == 0:
                                en_list = [new_list[dense_l_i + 1]]
                            if u != 0:
                                if en_list[0].size(2) != cat_features[u].shape[2]:
                                    m = 0
                                    padding_size = en_list[0].size(2) - cat_features[u].shape[2]
                                    if padding_size >= 0:
                                        if padding_size % 2 != 0:
                                            padding_size += 1
                                            m = -1
                                        padding_shape = (0, 0, 0, padding_size, 0, 0, 0, 0)
                                        expend = F.pad(cat_features[u], padding_shape, 'constant', 0)
                                        if m == 0:
                                            cat_features[u] = expend
                                        else:
                                            cat_features[u] = expend[:, :, :-1, :]
                                    else:
                                        padding_size = -padding_size
                                        if padding_size % 2 != 0:
                                            padding_size += 1
                                            m = -1
                                        padding_shape = (0, 0, 0, padding_size, 0, 0, 0, 0)
                                        expend = F.pad(en_list[0], padding_shape, 'constant', 0)
                                        if m == 0:
                                            en_list[0] = expend
                                        else:
                                            en_list[0] = expend[:, :, :-1, :]
                    cat_features = torch.cat(cat_features+en_list, dim=1)
                    dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[f"x_{depth_idx}_{dense_l_i}"](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
        dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](dense_x[f"x_{0}_{self.depth - 1}"])
        return dense_x[f"x_{0}_{self.depth}"]


def patch_first_conv(model, new_in_channels, default_in_channels=3, pretrained=True):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
            break
    weight = module.weight.detach()
    module.in_channels = new_in_channels
    if not pretrained:
        module.weight = nn.parameter.Parameter(
            torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        )
        module.reset_parameters()
    elif new_in_channels == 1:
        new_weight = weight.sum(1, keepdim=True)
        module.weight = nn.parameter.Parameter(new_weight)
    else:
        new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
        for i in range(new_in_channels):
            new_weight[:, i] = weight[:, i % default_in_channels]
        new_weight = new_weight * (default_in_channels / new_in_channels)
        module.weight = nn.parameter.Parameter(new_weight)


class EncoderMixin:
    """Add encoder functionality such as:
    - output channels specification of feature tensors (produced by encoder)
    - patching first convolution for arbitrary input channels
    """

    _output_stride = 32

    @property
    def out_channels(self):
        """Return channels dimensions for each tensor of forward output of encoder"""
        return self._out_channels[: self._depth + 1]
    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return
        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])
        patch_first_conv(model=self, new_in_channels=in_channels, pretrained=pretrained)


class EfficientNetEncoder(EfficientNet, EncoderMixin):
    def __init__(self, stage_idxs, out_channels, model_name, depth=5):
        blocks_args, global_params = get_model_params(model_name, override_params=None)
        super().__init__(blocks_args, global_params)
        self._stage_idxs = stage_idxs
        self._out_channels = out_channels
        self._depth = depth
        self._in_channels = 3
        del self._fc

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self._conv_stem, self._bn0, self._swish),
            self._blocks[: self._stage_idxs[0]],
            self._blocks[self._stage_idxs[0]: self._stage_idxs[1]],
            self._blocks[self._stage_idxs[1]: self._stage_idxs[2]],
            self._blocks[self._stage_idxs[2]:],
        ]
    def forward(self, x):
        stages = self.get_stages()
        block_number = 0.0
        drop_connect_rate = self._global_params.drop_connect_rate
        features = []
        for i in range(self._depth + 1):
            if i < 2:
                x = stages[i](x)
            else:
                for module in stages[i]:
                    drop_connect = drop_connect_rate * block_number / len(self._blocks)
                    block_number += 1.0
                    x = module(x, drop_connect)
            features.append(x)
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("_fc.bias", None)
        state_dict.pop("_fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)

def _get_pretrained_settings(encoder):
    pretrained_settings = {
        "imagenet": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "url": url_map[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
        "advprop": {
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "url": url_map_advprop[encoder],
            "input_space": "RGB",
            "input_range": [0, 1],
        },
    }
    return pretrained_settings


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    encoders = {
        "efficientnet-b7": {
            "encoder": EfficientNetEncoder,
            "pretrained_settings": _get_pretrained_settings("efficientnet-b7"),  #预训练配置
            "params": {
                "out_channels": (3, 64, 48, 80, 224, 640),
                "stage_idxs": (11, 18, 38, 55),
                "model_name": "efficientnet-b7",
            },
        },
    }
    Encoder = encoders[name]["encoder"]
    params = encoders[name]["params"]
    params.update(depth=depth)
    encoder = Encoder(**params)
    if weights is not None:
        settings = encoders[name]["pretrained_settings"][weights]
        encoder.load_state_dict(model_zoo.load_url(settings["url"]))
    encoder.set_in_channels(in_channels, pretrained=weights is not None)
    return encoder


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class FPT(nn.Module):
    def __init__(self, encoder_name, encoder_weights, in_channels, classes):
        super(FPT, self).__init__()
        self.encoder_depth = 5
        self.decoder_use_batchnorm = True,
        self.decoder_channels = (256, 128, 64, 32, 16)
        self.decoder_attention_type = None
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=self.encoder_depth,
            weights=encoder_weights,
        )
        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )
        self.decoder_1 = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=self.decoder_channels,
            n_blocks=self.encoder_depth,
            use_batchnorm=self.decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=self.decoder_attention_type,
        )
        self.segmentation_head_1 = SegmentationHead(
            in_channels=self.decoder_channels[-1],
            out_channels=classes,
            kernel_size=3,
        )


    def forward(self, x, distribution):
        img_H, img_W = x.shape[0], x.shape[1]
        features = self.encoder(x.permute(2, 0, 1).unsqueeze(0))
        decoder_output = self.decoder(*features)
        results = self.segmentation_head(decoder_output)
        ### center crop
        results = crop(results, img_H, img_W, 0, 0)
        if distribution == "beta":
            results = nn.Softplus()(results)
        decoder_output_1 = self.decoder_1(*features)
        results_1 = self.segmentation_head_1(decoder_output_1)
        std = crop(results_1, img_H, img_W, 0, 0)
        if distribution != "residual":
            std = nn.Softplus()(std)
        return results, std

def crop(data1, h, w, crop_h, crop_w):
    _, _, h1, w1 = data1.size()
    assert (h <= h1 and w <= w1)
    data = data1[:, :, crop_h:crop_h + h, crop_w:crop_w + w]
    return data
def cross_entropy_loss_RCF(prediction, labelef,std,ada):
    label = labelef.long()
    mask = label.float()
    num_positive = torch.sum((mask != 0).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()
    assert num_negative + num_positive == label.shape[0] * label.shape[1]*label.shape[2] * label.shape[3]
    mask[mask != 0] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    new_mask = mask * torch.exp(std * ada)
    cost = F.binary_cross_entropy(
        prediction, labelef, weight=new_mask.detach(), reduction='sum')
    return cost, mask

def Uncertainty_Aware(x,gt_src,dict):
    args = dict
    img_src = x
    temp_image = torch.from_numpy(img_src).float().to(args.gpu)
    meanUAED = FPT(encoder_name="efficientnet-b7", encoder_weights="imagenet", in_channels=3, classes=1).to(args.gpu)
    meanUAED_opt = optim.Adam(meanUAED.parameters(), lr=0.001, weight_decay=args.weight_decay)
    meanUAED.train()
    counter = 0
    label = gt_src
    label = torch.from_numpy(label).float().to(args.gpu)
    std_deviation = torch.std(label, dim=0, unbiased=False, keepdim=True)
    std_deviation_filled = std_deviation.expand_as(label)
    label_std = std_deviation_filled.cuda().unsqueeze(0).unsqueeze(0)
    label = label.unsqueeze(0).unsqueeze(0)
    for i in range(1):
        for k in range(int(img_src.shape[2] / 3)):
            image = img_src[:, :, 3 * k:3 * (k + 1)]
            image = torch.from_numpy(image).float().to(args.gpu)
            mean, std = meanUAED(image, distribution='gs')
            outputs_dist = Independent(Normal(loc=mean, scale=std + 0.001), 1)
            outputs = torch.sigmoid(outputs_dist.rsample())
            zmask = outputs >= 0.9
            counter += 1
            ada = (k + 1) / int(img_src.shape[2] / 3)
            bce_loss, mask = cross_entropy_loss_RCF(outputs, label, std, ada)
            std_loss = torch.sum((std - label_std) ** 2 * mask)
            loss_en = (bce_loss + std_loss * args.std_weight) / args.itersize
            meanUAED_opt.zero_grad()
            loss_en.backward()
            meanUAED_opt.step()
    masks = zmask.view(zmask.size(-2), zmask.size(-1),-1)
    labels = label.view(label.size(-2), label.size(-1),-1)
    dlabel = labels*masks
    dimage = temp_image*masks
    torch.cuda.empty_cache()
    return dimage,dlabel

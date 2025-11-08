
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# model
def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True) 
        x3 = self.conv3(x2)

        out = F.relu(x1+x3, inplace=True)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2, CLASS_NUM, patch_size, n_bands, embed_dim):
        super(D_Res_3d_CNN, self).__init__()
        self.n_bands = n_bands
        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(1,2,2),stride=(1,2,2), padding=(0,1,1))
        self.conv1 = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=(1,3,3), bias=False)
        self.patch_size = patch_size
        self.fc = nn.Linear(in_features=self._get_layer_size(), out_features=embed_dim, bias=False)
        self.classifier = nn.Linear(in_features=self._get_layer_size(), out_features=int(CLASS_NUM), bias=False)

    def _get_layer_size(self):
        with torch.no_grad():
            x = torch.zeros((1,1, self.n_bands,
                             self.patch_size, self.patch_size))
            x = self.block1(x)
            x = self.maxpool1(x)
            x = self.block2(x)
            x = self.maxpool2(x)
            x = self.conv1(x)
            x = x.view(x.shape[0],-1)
            s = x.size()[1]
        return s

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.conv1(x)
        x = x.view(x.shape[0],-1)
        y = self.classifier(x)
        proj = self.fc(x)
        return y, proj


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)
class AdaIN2d(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class TIFLM(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 inchannel,
                 vision_patch_size: int,
                 num_classes,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 device: 0,
                 n=16,
                 imdim=3
                 ):
        super().__init__()
        self.context_length = context_length
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.adain2_morph = AdaIN2d(10, 4)
        self.device = device
        self.visual = D_Res_3d_CNN(1,8,16,num_classes, vision_patch_size, inchannel, embed_dim)
        self.initialize_parameters()
        self.conv1 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=vision_patch_size/8, mode='bilinear', align_corners=True)
        kernelsize = 3
        stride = (kernelsize - 1) // 2
        self.conv2 = nn.Conv2d(4, n, 1, 1)  # 一个 1x1 卷积层，将3个通道转换为 n 个通道。
        self.conv3 = nn.Conv2d(n, imdim, 3, 1, stride)

    def initialize_parameters(self):  #用于初始化模型的参数。
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))  #
        mask.triu_(1)
        return mask


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, mode):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


    def forward(self, image, text=None, label=None, text_queue_1=None, text_queue_2=None):
        imgage_prob, image_features = self.encode_image(image, mode='train')
        if self.training:
            if text!=None:
                temp_features = image_features.view(image_features.size(0), round(image_features.size(1) ** (1 / 3)),
                                                    round(image_features.size(1) ** (1 / 3)),
                                                    round(image_features.size(1) ** (1 / 3)))

                temp_features = self.conv1(temp_features)
                temp_features = self.upsample(temp_features)
                z = torch.randn(len(image_features), 10).to(self.device)
                text_image_features = self.adain2_morph(temp_features, z)
                text_image_features = self.conv2(text_image_features)
                text_image_features = torch.sigmoid(self.conv3(text_image_features))
                text_features = self.encode_text(text)
                text_features_q1 = self.encode_text(text_queue_1)
                text_features_q2 = self.encode_text(text_queue_2)
                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)
                # cosine similarity as logits
                logit_scale = self.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logit_scale * text_features @ image_features.t()
                loss_img = F.cross_entropy(logits_per_image, label.long())
                loss_text = F.cross_entropy(logits_per_text, label.long())
                loss_clip = (loss_img + loss_text)/2
                text_features_q1 = text_features_q1 / text_features_q1.norm(dim=1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features_q1.t()
                logits_per_text = logit_scale * text_features_q1 @ image_features.t()
                loss_img = F.cross_entropy(logits_per_image, label.long())
                loss_text = F.cross_entropy(logits_per_text, label.long())
                loss_q1 = (loss_img + loss_text)/2
                text_features_q2 = text_features_q2 / text_features_q2.norm(dim=1, keepdim=True)
                logit_scale = self.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features_q2.t()
                logits_per_text = logit_scale * text_features_q2 @ image_features.t()
                loss_img = F.cross_entropy(logits_per_image, label.long())
                loss_text = F.cross_entropy(logits_per_text, label.long())
                loss_q2 = (loss_img + loss_text)/2
                return loss_clip, (loss_q1+loss_q2)/2, imgage_prob,text_image_features
            else:
                return imgage_prob
        else:
            return imgage_prob


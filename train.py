from __future__ import print_function
import argparse
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from model.CLIP import TIFLM
from model.UAM import Uncertainty_Aware
from model.DFLA import ULE
from model.CBDM import Shapley


import numpy as np
from utils_HSI import sample_gt, metrics, get_device, seed_worker
from datasets import get_dataset, HyperX, data_prefetcher
from datetime import datetime
import os
import torch.utils.data as data

from sklearn.metrics import classification_report
import clip
import time
parser = argparse.ArgumentParser(description='ULDGN')
parser.add_argument('--save_path', type=str, default="./results/",
                    help='the path to save the model')
parser.add_argument('--data_path', type=str, default='./datasets/Houston/')
parser.add_argument('--source_name', type=str, default='Houston13',
                    help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='Houston18',
                    help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")

# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float, default=1e-3,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--lambda_1', type=float, default=10,
                    help="Regularization parameter, balancing the alignment loss.")
group_train.add_argument('--alpha', type=float, default=0.7,
                    help="Regularization parameter, controlling the contribution of both coarse-and fine-grained linguistic features.")
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')

group_train.add_argument('--batch_size', type=int, default=256,
                    help="Batch size (optional, if absent will be set by the model")

parser.add_argument('--seed', type=int, default=30014, metavar='S',
                    help='random seed ')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=1e-4,
                    help='the L2  weight decay')
group_train.add_argument('--lambda_2', type=float, default=0.01)
group_train.add_argument('--lambda_3', type=float, default=2)

parser.add_argument('--num_epoch', type=int, default=200,
                    help='the number of epoch')
parser.add_argument('--num_trials', type=int, default=1,
                    help='the number of epoch')
parser.add_argument('--training_sample_ratio', type=float, default=0.8,
                    help='training sample ratio')
parser.add_argument('--re_ratio', type=int, default=1,
                    help='multiple of of data augmentation')

group_train.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--std_weight', default=1, type=float,help='weight for std loss')
parser.add_argument('--itersize', default=3, type=int,
                    metavar='IS', help='iter size')
parser.add_argument('--idleness_epoch', type=int, default=1)
parser.add_argument('--d_se', type=int, default=32)

group_train.add_argument('--alpha1', type=float, default=1)
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=False,
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',default=False,
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',default=False,
                    help="Random mixes between spectra")

args = parser.parse_args()
DEVICE = get_device(args.gpu)


def Divergent_sample_processing(img_src,gt_src, args):
    file_path = args.data_path + '/'+args.source_name+'_df_image.npy'
    if os.path.isfile(file_path):
        df_image = np.load(args.data_path + '/'+args.source_name+'_df_image.npy')
        df_dlabel = np.load(args.data_path + '/'+args.source_name+'_df_dlabel.npy')
        unique_values = np.unique(df_dlabel)
        df_numclass = len(unique_values) - 1
    else:
        df_image, df_dlabel = Uncertainty_Aware(img_src, gt_src, args)
        df_dlabel = df_dlabel.squeeze(-1)
        df_numclass = len(df_dlabel.unique()) - 1  # 除去0
        df_image = df_image.cpu().numpy()
        df_dlabel = df_dlabel.cpu().numpy()
        np.save(args.data_path + '/'+args.source_name+'_df_image.npy', df_image)
        np.save(args.data_path + '/'+args.source_name+'_df_dlabel.npy', df_dlabel)
    return df_image, df_dlabel, df_numclass
def Divergent_feature_extraction(df_train_loader,df_net,df_opt,cls_criterion,hyperparams,):
    df_feature = []
    for i, (x, y) in enumerate(df_train_loader):
        df_data, df_label = x.to(args.gpu), y.to(args.gpu)
        for k in range(len(df_label.unique())):
            mask = df_label == df_label.unique()[k]
            df_label[mask] = k
        df_pred, df_proj = df_net(df_data)
        df_loss = cls_criterion(df_pred, df_label.long())
        df_opt.zero_grad()
        df_loss.backward()
        df_opt.step()
        df_feature.append(df_proj.detach().cpu().numpy())
    df_feature = torch.tensor(np.concatenate(df_feature))
    n, c, h, w = df_feature.shape
    df_feature = df_feature.transpose(0, 1)
    conv_layer = nn.Conv2d(in_channels=n, out_channels=hyperparams['batch_size'], kernel_size=1, stride=1, padding=0)
    df_feature = conv_layer(df_feature.data)
    df_feature = df_feature.transpose(0, 1)
    conv1 = torch.nn.Conv2d(in_channels=c, out_channels=hyperparams['n_bands'], kernel_size=1)
    output_tensor = conv1(df_feature)
    df_feature = F.interpolate(output_tensor, size=(args.patch_size, args.patch_size), mode='bilinear', align_corners=False)
    N, C, H, W = df_feature.size()
    x = df_feature.view(N, C, -1)
    mean = x.mean(1, keepdim=True)
    var = x.var(1, keepdim=True)
    eps = 1e-5
    x = (x - mean) / (var + eps).sqrt()
    idx_swap = torch.randperm(N)
    alpha = torch.rand(N, 1, 1)
    mean = alpha * mean + (1 - alpha) * mean[idx_swap]
    var = alpha * var + (1 - alpha) * var[idx_swap]
    x = x * (var + eps).sqrt() + mean
    df_feature = x.view(N, C, H, W).to(args.gpu)
    torch.cuda.empty_cache()
    return df_feature

def execute_modulation(model, device, optimizer, image, spec,label,drop=None):
    criterion = nn.CrossEntropyLoss()
    model.train()
    optimizer.zero_grad()
    image = image.to(args.gpu).detach()
    spec = spec.to(args.gpu).detach()
    label = label.to(args.gpu).detach()
    if drop is not None:
        drop = drop.to(device)
    a, v, out = model(spec, image, drop,device=device)
    inter_outputs = out.clone()
    merge_loss1 = criterion(inter_outputs, label.long())
    optimizer.zero_grad()
    merge_loss1.backward()
    optimizer.step()
    torch.cuda.empty_cache()
    contribution = {}
    softmax = nn.Softmax(dim=1)
    cona = 0.0
    conv = 0.0
    with torch.no_grad():
        model.eval()
        image = image.to(device)
        spec = spec.to(device)
        a, v, out = model(spec.float(), image.float(),device=device)
        out_v = model.module.exec_drop(a, v, drop="audio")
        out_a = model.module.exec_drop(a, v, drop="visual")
        prediction = softmax(out)
        pred_v = softmax(out_v)
        pred_a = softmax(out_a)
        for i in range(len(label)):
            all = prediction[i].cpu().data.numpy()
            index_all = np.argmax(all)
            v = pred_v[i].cpu().data.numpy()
            index_v = np.argmax(v)
            a = pred_a[i].cpu().data.numpy()
            index_a = np.argmax(a)
            value_all = 0.0
            value_a = 0.0
            value_v = 0.0
            if index_all == label[i]:
                value_all = 2.0
            if index_v == label[i]:
                value_v = 1.0
            if index_a == label[i]:
                value_a = 1.0
            contrib_a = (value_a + value_all - value_v) / 2.0
            contrib_v = (value_v + value_all - value_a) / 2.0
            cona += contrib_a
            conv += contrib_v
            contribution[i] = (contrib_a, contrib_v)
    a_mean, v_mean = cona,conv
    a_weight, v_weight = a_mean / (a_mean + v_mean), v_mean / (a_mean + v_mean)
    part_cona = 0.0
    part_conv = 0.0
    num = int(len(label) * args.training_sample_ratio)
    choice = np.random.choice(len(label), num)
    for i in choice:
        contri_a, contri_v = contribution[i]
        part_cona += contri_a
        part_conv += contri_v
    part_cona /= num
    part_conv /= num
    gap_a = 1.0 - part_cona
    gap_v = 1.0 - part_conv
    a_std, v_std = gap_a, gap_v

    part_difference = abs(gap_a - gap_v) / 3 * 2 * args.alpha1

    torch.cuda.empty_cache()

    return merge_loss1,a_weight,v_weight,a_std,v_std,part_difference

def Data_fusion(df_feature, x_ED, x, y, num_classes, args, epoch, flagepoch, idleness_flag, static_epoch,epoch_flag):
    merge_model = Shapley(num_classes=num_classes).to(args.gpu)
    merge_model = torch.nn.DataParallel(merge_model, device_ids=[args.gpu])
    merge_model.to(args.gpu)
    merge_opt = optim.Adam(
        merge_model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-4,
        amsgrad=False,
    )
    if df_feature.shape[0] != x_ED.shape[0]:
        shape = df_feature.shape[0] - x_ED.shape[0]
        padded_tensor1 = torch.zeros(df_feature.shape, dtype=df_feature.dtype, device=df_feature.device)
        padded_tensor1[:x_ED.shape[0]] = x_ED
        x_ED = padded_tensor1
        padded_tensor2 = torch.zeros(df_feature.shape, dtype=df_feature.dtype, device=df_feature.device)
        padded_tensor2[:x.shape[0]] = x
        x = padded_tensor2
        y = F.pad(y, (0, shape), 'constant', 0)
    merge_loss, ED_mean, df_mean, ED_std, df_std, part_difference = execute_modulation(model=merge_model,
                                                                                       device=args.gpu,
                                                                                       optimizer=merge_opt,
                                                                                       image=df_feature, spec=x_ED,
                                                                                       label=y)
    shapes = (len(df_feature), 1, 1, 1)
    if ED_mean > df_mean:
        idleness_rand = torch.normal(mean=ED_mean, std=ED_std, size=shapes).to(args.gpu)
        anxiety_rand = torch.normal(mean=df_mean, std=df_std, size=shapes).to(args.gpu)
    else:
        idleness_rand = torch.normal(mean=df_mean, std=df_std, size=shapes).to(args.gpu)
        anxiety_rand = torch.normal(mean=ED_mean, std=ED_std, size=shapes).to(args.gpu)
    if epoch < args.num_epoch * 0.2:
        x_ID = anxiety_rand * df_feature + idleness_rand * x_ED
        static_epoch = args.num_epoch * 0.2  # 36
    elif args.num_epoch * 0.2 <= epoch < args.num_epoch * 0.7:
        if epoch == int(static_epoch + flagepoch):
            x_ID = idleness_rand * df_feature + anxiety_rand * x_ED
            epoch_flag = True
        else:
            x_ID = anxiety_rand * df_feature + idleness_rand * x_ED
            epoch_flag = False
    else:
        if idleness_flag:
            x_ID = anxiety_rand * df_feature + idleness_rand * x_ED
        else:
            x_ID = idleness_rand * df_feature + anxiety_rand * x_ED
    alpha = np.random.beta(0.6, 0.6)
    x_mix = alpha * x_ID + (1 - alpha) * x
    torch.cuda.empty_cache()
    return x_ID,x_mix,x,y,x_ED,static_epoch,epoch_flag


def train(epoch, model, num_epoch, label_name, label_queue,static_epoch,epoch_flag):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch - 1) / num_epoch), 0.75)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if (epoch-1)%10==0:
        print('learning rate{: .4f}'.format(LEARNING_RATE) )
    CNN_correct= 0
    iter_source = iter(train_loader)
    num_iter = len_src_loader
    for i in range(1, num_iter):
        model.train()
        data_src, label_src = next(iter_source)
        data_src, label_src = data_src.to(DEVICE), label_src.to(DEVICE)
        label_src = label_src - 1
        optimizer.zero_grad()
        text = torch.cat([clip.tokenize(f'A hyperspectral image of {label_name[int(k)]}').to(k.device) for k in label_src])
        text_queue_1 = [label_queue[label_name[int(k)]][0] for k in label_src]
        text_queue_2 = [label_queue[label_name[int(k)]][1] for k in label_src]
        text_queue_1 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_1])
        text_queue_2 = torch.cat([clip.tokenize(k).to(text.device) for k in text_queue_2])
        loss_coarse, loss_fine, label_src_pred,x_ED = model(data_src, text, label_src, text_queue_1=text_queue_1, text_queue_2=text_queue_2)
        loss_cls = F.nll_loss(F.log_softmax(label_src_pred, dim=1), label_src.long())
        loss = args.lambda_3*loss_cls + args.lambda_1*((1-args.alpha)*loss_coarse + args.alpha*loss_fine)
        loss.backward()
        optimizer.step()
        x_ID, x_mix, x_fill, y_fill, x_ED_fill, static_epoch, epoch_flag = Data_fusion(df_feature, x_ED, data_src, label_src,
                                                                                       num_classes, args, epoch,
                                                                                       flagepoch, idleness_flag,
                                                                                       static_epoch, epoch_flag)
        x, y, x_ED = x_fill, y_fill, x_ED_fill
        model.eval()

        predict11 = model(x.detach())
        predict21  = model(x_ED.detach())
        predict31  = model(x_ID.detach())
        predict41  = model(x_mix.detach())

        loss_aug1 = cls_criterion(predict11, y.long())
        loss_aug2 = cls_criterion(predict21, y.long())
        loss_aug3 = cls_criterion(predict31, y.long())
        loss_aug4 = cls_criterion(predict41, y.long())

        prob11 = torch.softmax(predict11, dim=1)
        prob21 = torch.softmax(predict21, dim=1)
        prob31 = torch.softmax(predict31, dim=1)
        prob41 = torch.softmax(predict41, dim=1)

        criterion = torch.nn.KLDivLoss(reduction='batchmean')
        loss_kl1 = criterion(prob11, prob41)+ criterion(prob21, prob31)
        loss_min = args.lambda_2*loss_kl1 + loss_aug1 + loss_aug2 + loss_aug3 + loss_aug4
        loss_min.backward()
        optimizer.step()
        pred = predict11.data.max(1)[1]
        CNN_correct += pred.eq(label_src.data.view_as(pred)).cpu().sum()
        torch.cuda.empty_cache()
        if i % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format( epoch, i * len(data_src), len_src_dataset, 100. * i / len_src_loader))
            print('loss: {:.6f},  loss_cls: {:.6f},  loss_coarse: {:.6f}, loss_fine: {:.6f}, loss_kl1: {:.6f}, loss_min: {:.6f}'.format(
            loss.item(), loss_cls.item(), loss_coarse.item(), loss_fine.item(), loss_kl1.item(), loss_min.item()))
    CCN_acc = CNN_correct.item() / len_src_dataset
    print('[epoch: {:4}]  Train Accuracy: {:.4f} | train sample number: {:6}'.format(epoch, CCN_acc, len_src_dataset))
    torch.cuda.empty_cache()
    return model, CCN_acc,static_epoch,epoch_flag

def test(model, label_name):
    model.eval()
    loss = 0
    correct = 0
    pred_list, label_list,feature = [], [],[]

    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            label = label - 1
            label_src_pred = model(data)
            pred = label_src_pred.data.max(1)[1]
            pred_list.append(pred.cpu().numpy())
            feature.append(label_src_pred.detach().cpu().numpy())
            label_list.append(label.cpu().numpy())
            loss += F.nll_loss(F.log_softmax(label_src_pred, dim = 1), label.long()).item()
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
        loss /= len_tar_loader

        print('Average test loss: {:.4f}, test Accuracy: {}/{} ({:.2f}%), | test sample number: {:6}\n'.format(
            loss, correct, len_tar_dataset, 100. * correct / len_tar_dataset, len_tar_dataset))


    return correct, correct.item() / len_tar_dataset, pred_list, label_list
if __name__ == '__main__':
    hyperparams = vars(args)
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name+'to'+args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_pt'+str(args.patch_size)+'_bs'+str(args.batch_size)+'_'+time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    args.save_path = os.path.join(args.save_path)
    acc_test_list, acc_maxval_test_list = np.zeros([args.num_trials,1]), np.zeros([args.num_trials,1])
    seed_worker(args.seed)

    img_src, gt_src, LABEL_VALUES_src, LABEL_QUEUE, IGNORED_LABELS = get_dataset(args.source_name,
                                                            args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, LABEL_QUEUE, IGNORED_LABELS = get_dataset(args.target_name,
                                                            args.data_path)
    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    training_sample_tar_ratio = args.training_sample_ratio*args.re_ratio*sample_num_src/sample_num_tar

    num_classes=gt_src.max()
    N_BANDS = img_src.shape[-1]

    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 
                        'device': DEVICE, 'center_pixel': False, 'supervision': 'full'})
    hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)   #更新超参数



    df_image, df_dlabel, df_numclass = Divergent_sample_processing(img_src, gt_src, args)

    r = int(hyperparams['patch_size']/2)+1
    img_src=np.pad(img_src,((r,r),(r,r),(0,0)),'symmetric')
    img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
    df_image = np.pad(df_image, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src=np.pad(gt_src,((r,r),(r,r)),'constant',constant_values=(0,0))
    gt_tar=np.pad(gt_tar,((r,r),(r,r)),'constant',constant_values=(0,0))
    df_dlabel = np.pad(df_dlabel, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    train_gt_src, _, training_set, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
    train_gt_df, _, _, _ = sample_gt(df_dlabel, 1, mode='random')
    test_gt_tar, _, tesing_set, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src

    for i in range(args.re_ratio-1):
        img_src_con = np.concatenate((img_src_con,img_src))
        train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
    
    hyperparams_train = hyperparams.copy()
    hyperparams_train.update({'flip_augmentation': True, 'radiation_augmentation': True, 'mixture_augmentation': False})


    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = data.DataLoader(train_dataset,
                                    batch_size=hyperparams['batch_size'],
                                    pin_memory=True,
                                    worker_init_fn=seed_worker,
                                    generator=g,
                                    shuffle=True)
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)

    df_train_dataset = HyperX(df_image, df_dlabel, **hyperparams_train)
    df_train_loader = torch.utils.data.DataLoader(df_train_dataset, batch_size=hyperparams['batch_size'], pin_memory=True,
                                      worker_init_fn=seed_worker, generator=g)

    test_loader = data.DataLoader(test_dataset,
                                    pin_memory=True,
                                    # worker_init_fn=seed_worker,
                                    # generator=g,
                                    batch_size=hyperparams['batch_size'])                          
    len_src_loader = len(train_loader)
    len_src_dataset = len(train_loader.dataset)
    len_tar_dataset = len(test_loader.dataset)
    len_tar_loader = len(test_loader)

    print(hyperparams)
    print("train samples :",len_src_dataset)

    correct, acc = 0, 0
    flagepoch = 13
    epoch_flag =False
    idleness_flag =  True
    static_epoch = 0
    df_net = ULE(in_channels=N_BANDS, num_classes=df_numclass).to(args.gpu)
    df_opt = torch.optim.Adam(df_net.parameters(), lr=args.lr)
    pretrained_dict = torch.jit.load('./ViT-B-32.pt', map_location="cpu").state_dict()
    embed_dim = pretrained_dict["text_projection"].shape[1]
    context_length = pretrained_dict["positional_embedding"].shape[0]
    vocab_size = pretrained_dict["token_embedding.weight"].shape[0]
    transformer_width = pretrained_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = 3
    model = TIFLM(embed_dim,
        img_src.shape[-1], hyperparams['patch_size'], gt_src.max(),
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,device=args.gpu,n=args.d_se,imdim=N_BANDS).to(DEVICE)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in pretrained_dict:
            del pretrained_dict[key]
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'visual' not in k.split('.')}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params/(1024*1024):.2f}M training parameters.')
    cls_criterion = torch.nn.CrossEntropyLoss()

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    for epoch in range(1, args.num_epoch + 1):
        t1 = time.time()
        df_net.train()
        df_feature=Divergent_feature_extraction(df_train_loader,df_net, df_opt, cls_criterion, hyperparams)
        if epoch % args.idleness_epoch == 0:
            idleness_flag =  not idleness_flag
        model, CCN_train_acc,static_epoch,epoch_flag = train(epoch, model, args.num_epoch, LABEL_VALUES_src, LABEL_QUEUE,static_epoch,epoch_flag)
        t2 = time.time()
        print('train epoch time:', t2-t1)
        if epoch_flag:
            static_epoch = epoch
            flagepoch = flagepoch - 1
        torch.cuda.empty_cache()
        t_correct, CCN_test_acc, pred, label = test(model, LABEL_VALUES_src)
        print('accuracy{: .2f}%\n'.format(100. * t_correct / len_tar_dataset))
        if t_correct > correct:

            correct = t_correct
            acc = CCN_test_acc
            results = metrics(np.concatenate(pred), np.concatenate(label), ignored_labels=hyperparams['ignored_labels'], n_classes=int(gt_src.max()))
            print(results['Confusion_matrix'], '\n', 'TPR:', np.round(results['TPR'] * 100, 2), '\n', 'OA:',
                  results['Accuracy'], 'F1scores:', results['F1_scores'], 'kappa:',
                  results['Kappa'], 'AA:',
                  results['AA'])


        max_accuracy = 100. * correct / len_tar_dataset
        print('source: {} to target: {} max correct: {} max accuracy{: .2f}%\n'.format(
            args.source_name, args.target_name, correct, 100. * correct / len_tar_dataset ))
        



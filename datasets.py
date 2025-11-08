# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import torch.utils.data
import os
from tqdm import tqdm
from scipy import io
from scipy.linalg import sqrtm
try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from utils_HSI import open_file
import matplotlib.pyplot as plt
import random

DATASETS_CONFIG = {
        'Houston13': {
            'img': 'Houston13.mat',
            'gt': 'Houston13_7gt.mat',
            },
        'Houston18': {
            'img': 'Houston18.mat',
            'gt': 'Houston18_7gt.mat',
            },
        'paviaU': {
            'img': 'paviaU.mat',
            'gt': 'paviaU_7gt.mat',
            },
        'paviaC': {
            'img': 'paviaC.mat',
            'gt': 'paviaC_7gt.mat',
            },
        'Dioni': {
            'img': 'Dioni.mat',
            'gt': 'Dioni_gt_out68.mat',
            },
    'Loukia': {
        'img': 'Loukia.mat',
        'gt': 'Loukia_gt_out68.mat',
    }
    }

try:
    from custom_datasets import CUSTOM_DATASETS_CONFIG
    DATASETS_CONFIG.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset(dataset_name, target_folder="./", datasets=DATASETS_CONFIG):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None
    
    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))    #数据集名称不在配置中

    dataset = datasets[dataset_name]#得到数据集文件名

    folder = target_folder# + datasets[dataset_name].get('folder', dataset_name + '/')
    if dataset.get('download', False):
        # Download the dataset if is not present
        if not os.path.isdir(folder):
            os.mkdir(folder)
        for url in datasets[dataset_name]['urls']:
            # download the files
            filename = url.split('/')[-1]
            if not os.path.exists(folder + filename):
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                          desc="Downloading {}".format(filename)) as t:
                    urlretrieve(url, filename=folder + filename,
                                     reporthook=t.update_to)
    elif not os.path.isdir(folder):
       print("WARNING: {} is not downloadable.".format(dataset_name))

    if dataset_name == 'Houston13':
        # Load the image
        img = np.asarray(open_file(folder + 'Houston13.mat')['ori_data']).transpose(1, 2, 0)   #图像信息

        rgb_bands = [13,20,33]

        gt = np.asarray(open_file(folder + 'Houston13_7gt.mat')['map'])  #标签值

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]  #标签信息
        label_queue = {"grass healthy":['The grass healthy is next to the road','The grass healthy is dark green'],
                    "grass stressed":['The grass stressed is next to the road','The grass stressed is pale green'],
                    "trees":['The trees beside road','The trees appear as small circles'],
                    "water":['The water has a smooth surface','The water appears dark blue or black'],
                    "residential buildings":['Residential buildings are densely packed','Residential buildings appear as small blocks'],
                    "non-residential buildings":['The shapes of the non-residential buildings are inconsistent','Non-residential buildings appear as large blocks'],
                    "road":['Trees grew along the road','The road appear as elongated strip shape']}   #粗粒度 细粒度文本

        ignored_labels = [0]   #忽略标签

    elif dataset_name == 'Houston18':
        # Load the image
        img = np.asarray(open_file(folder + 'Houston18.mat')['ori_data']).transpose(1, 2, 0)

        rgb_bands = [13,20,33]

        gt = np.asarray(open_file(folder + 'Houston18_7gt.mat')['map'])

        label_values = ["grass healthy", "grass stressed", "trees",
                        "water", "residential buildings",
                        "non-residential buildings", "road"]
        label_queue = {"grass healthy":['The grass healthy is next to the road','The grass healthy is dark green'],
                    "grass stressed":['The grass stressed is next to the road','The grass stressed is pale green'],
                    "trees":['The trees grew along the road','The trees appear as small circles'],
                    "water":['The water has a smooth surface','The water appears dark blue or black'],
                    "residential buildings":['Residential buildings are densely packed','Residential buildings appear as small blocks'],
                    "non-residential buildings":['The shapes of the non-residential buildings are inconsistent','Non-residential buildings appear as large blocks'],
                    "road":['Trees on the roadside','The road appear as elongated strip shape']}
        ignored_labels = [0]

    elif dataset_name == 'paviaU':

        # Load the image
        img = np.asarray(io.loadmat(folder + 'paviaU.mat')['ori_data'])

        rgb_bands = [20,30,30]

        gt = np.asarray(io.loadmat(folder + 'paviaU_7gt.mat')['map'])

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        label_queue = {"tree":['The trees beside road','The trees appear as small circles'],
                    "asphalt":['Trees grew along the asphalt road','The asphalt road appear as elongated strip shape'],
                    "brick":['Brick is a kind of road material','Bricks are generally arranged in strips'],
                    "bitumen":['Bitumen is a material for building surfaces','Bitumen is a widely used waterproof material'],
                    "shadow":['The shadows next to buildings','The shadow appears black'],
                    "meadow":['The surface of the meadow is green with grass','The surface of the meadow is green with grass'],
                    "bare soil":['No grass on the surface of Bare soil','The bare soil appears grayish-black color']}

        ignored_labels = [0]
    elif dataset_name == 'paviaC':
        # Load the image
        img = np.asarray(io.loadmat(folder + 'paviaC.mat')['ori_data'])

        rgb_bands = [20,30,30]

        gt = np.asarray(io.loadmat(folder + 'paviaC_7gt.mat')['map'])

        label_values = ["tree", "asphalt", "brick",
                        "bitumen", "shadow", 'meadow', 'bare soil']

        label_queue = {"tree":['The trees beside road','The trees appear as small circles'],
                    "asphalt":['Trees grew along the asphalt road','The asphalt road appear as elongated strip shape'],
                    "brick":['Brick is a kind of road material','Bricks are generally arranged in strips'],
                    "bitumen":['Bitumen is a material for building surfaces','Bitumen is a widely used waterproof material'],
                    "shadow":['The shadows next to buildings','The shadow appears black'],
                    "meadow":['The surface of the meadow is green with grass','The surface of the meadow is green with grass'],
                    "bare soil":['No grass on the surface of bare soil','The bare soil appears grayish-black color']}
                    
        ignored_labels = [0]
    elif dataset_name == 'Dioni':
        # Load the image

        rgb_bands = [23, 11, 6]


        img =io.loadmat(folder +'Dioni.mat')['ori_data']
        gt = io.loadmat(folder + 'Dioni_gt_out68.mat')['map']

        label_values = ["Dense urban fabric", "Mineral extraction sites", "Non-irrigated arable land",
                        "Fruit trees", "Olive groves", "Coniferous forest", "Dense sclerophyllous vegetation","Sparse sclerophyllous vegetation","Sparcely vegetated areas","Rocks and sand","Water","Coastal Water"]
        label_queue = {
            "Dense urban fabric": ['The dense urban fabric consists of closely packed buildings',
                                   'Dense urban fabric appears as a large block structure'],
            "Mineral extraction sites": ['Mineral extraction sites are characterized by excavation and bare soil',
                                         'These sites often appear as irregular patches with different textures'],
            "Non-irrigated arable land": ['Non-irrigated arable land lacks water and shows dry soil',
                                          'The land appears as dry, cracked earth with sparse vegetation'],
            "Fruit trees": ['Fruit trees are typically found in orchards with distinct tree crowns',
                            'These trees appear as small circles with round canopies'],
            "Olive groves": ['Olive groves consist of uniformly shaped olive trees',
                             'Olive groves appear as a pattern of small, dense trees'],
            "Coniferous forest": ['Coniferous forests are composed of conical-shaped evergreen trees',
                                  'These forests appear dark green with a sharp outline'],
            "Dense sclerophyllous vegetation": [
                'Dense sclerophyllous vegetation has hard-leaved, drought-resistant plants',
                'This vegetation appears as thick, dark green clusters'],
            "Sparse sclerophyllous vegetation": [
                'Sparse sclerophyllous vegetation consists of scattered, hard-leaved plants',
                'These plants appear as isolated dark green patches'],
            "Sparcely vegetated areas": ['Sparcely vegetated areas have a low density of vegetation',
                                         'These areas appear as light green or brown with patches of soil'],
            "Rocks and sand": ['Rocks and sand areas are characterized by the presence of rocks and sandy soil',
                               'These areas appear as grey or light brown with a rough texture'],
            "Water": ['Water bodies are smooth surfaces that reflect light',
                      'Water appears as dark blue or black with a reflective surface'],
            "Coastal Water": ['Coastal water is the water near the shore, influenced by the land',
                              'Coastal water appears as a gradient from dark to lighter blue near the shore']
        }

        ignored_labels = [0]
    elif dataset_name == 'Loukia':
        # Load the image

        img = io.loadmat(folder + 'Loukia.mat')['ori_data']
        gt = io.loadmat(folder + 'Loukia_gt_out68.mat')['map']

        label_values = ["Dense urban fabric", "Mineral extraction sites", "Non-irrigated arable land",
                        "Fruit trees", "Olive groves", "Coniferous forest", "Dense sclerophyllous vegetation","Sparse sclerophyllous vegetation","Sparcely vegetated areas","Rocks and sand","Water","Coastal Water"]
        label_queue = {
            "Dense urban fabric": ['The dense urban fabric consists of closely packed buildings',
                                   'Dense urban fabric appears as a large block structure'],
            "Mineral extraction sites": ['Mineral extraction sites are characterized by excavation and bare soil',
                                         'These sites often appear as irregular patches with different textures'],
            "Non-irrigated arable land": ['Non-irrigated arable land lacks water and shows dry soil',
                                          'The land appears as dry, cracked earth with sparse vegetation'],
            "Fruit trees": ['Fruit trees are typically found in orchards with distinct tree crowns',
                            'These trees appear as small circles with round canopies'],
            "Olive groves": ['Olive groves consist of uniformly shaped olive trees',
                             'Olive groves appear as a pattern of small, dense trees'],
            "Coniferous forest": ['Coniferous forests are composed of conical-shaped evergreen trees',
                                  'These forests appear dark green with a sharp outline'],
            "Dense sclerophyllous vegetation": [
                'Dense sclerophyllous vegetation has hard-leaved, drought-resistant plants',
                'This vegetation appears as thick, dark green clusters'],
            "Sparse sclerophyllous vegetation": [
                'Sparse sclerophyllous vegetation consists of scattered, hard-leaved plants',
                'These plants appear as isolated dark green patches'],
            "Sparcely vegetated areas": ['Sparcely vegetated areas have a low density of vegetation',
                                         'These areas appear as light green or brown with patches of soil'],
            "Rocks and sand": ['Rocks and sand areas are characterized by the presence of rocks and sandy soil',
                               'These areas appear as grey or light brown with a rough texture'],
            "Water": ['Water bodies are smooth surfaces that reflect light',
                      'Water appears as dark blue or black with a reflective surface'],
            "Coastal Water": ['Coastal water is the water near the shore, influenced by the land',
                              'Coastal water appears as a gradient from dark to lighter blue near the shore']
        }
        ignored_labels = [0]

    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))   #检查数组是否为NaN  对img最后一个轴进行求和
    if np.count_nonzero(nan_mask) > 0:  #检查nan_mask中的非零元素的个数
       print("Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)  #对非零元素清除    去除hsi中的无效元素

    ignored_labels = list(set(ignored_labels))#去除无效元素中的重复标签，然后将忽略标签转化为列表形式
    # Normalization
    img = np.asarray(img, dtype='float32')   #将原img数组转化为float32格式的数组
    
    m, n, d = img.shape[0], img.shape[1], img.shape[2]   #获取img数组3个维度的大小

    #数据归一化
    img = img.reshape((m*n,-1))#将img数组重塑为一个二维数组，其中每行是原始数组中的一个通道，并且将通道的所有像素值放在同一行中。
    img = img/img.max()#数组img中的每个元素除以数组中的最大值，以进行归一化，使其范围在0到1之间。
    img_temp = np.sqrt(np.asarray((img**2).sum(1)))#计算每行的平方和的平方根，并将结果保存在img_temp中。
    img_temp = np.expand_dims(img_temp,axis=1)#将img_temp数组的维度从一维扩展为二维，并将其形状调整为(m*n, 1)，以便后续的广播操作。
    img_temp = img_temp.repeat(d,axis=1)   #使用np.repeat()函数将img_temp中的每个元素沿着第二个维度复制d次，以便后续的元素级别的除法运算。   将图像数据扩充到每个光谱波段上
    img_temp[img_temp==0]=1  #将img_temp数组中值为0的元素替换为1，避免除法中的零除错误。
    img = img/img_temp  #将数组img中的每个元素除以对应位置上的img_temp数组中的元素，进行归一化。
    img = np.reshape(img,(m,n,-1))  #将归一化后的数组img重新调整为原始形状(m,n,-1)，恢复为一个三维数组，其中最后一个维度的大小与原始数组的通道数相同

    # return img, gt, label_values, ignored_labels, rgb_bands, palette
    return img, gt, label_values, label_queue, ignored_labels


class HyperX(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, transform=None, **hyperparams):
        """
        Args:
            data: 3D hyperspectral image
            gt: 2D array of labels
            patch_size: int, size of the spatial neighbourhood
            center_pixel: bool, set to True to consider only the label of the
                          center pixel
            data_augmentation: bool, set to True to perform random flips
            supervision: 'full' or 'semi' supervised algorithms
        """
        super(HyperX, self).__init__()
        self.transform = transform
        self.data = data
        self.label = gt
        self.patch_size = hyperparams['patch_size']
        self.ignored_labels = set(hyperparams['ignored_labels'])
        self.flip_augmentation = hyperparams['flip_augmentation']
        self.radiation_augmentation = hyperparams['radiation_augmentation'] 
        self.mixture_augmentation = hyperparams['mixture_augmentation'] 
        self.center_pixel = hyperparams['center_pixel']
        supervision = hyperparams['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.indices = np.array([(x,y) for x,y in zip(x_pos, y_pos) if x > p and x < data.shape[0] - p and y > p and y < data.shape[1] - p])
        self.labels = [self.label[x,y] for x,y in self.indices]
        
        # state = np.random.get_state()
        # np.random.shuffle(self.indices)
        # np.random.set_state(state)
        # np.random.shuffle(self.labels)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1/25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for  idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert(self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x,y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y = self.indices[i]
        x1, y1 = x - self.patch_size // 2, y - self.patch_size // 2
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size

        data = self.data[x1:x2, y1:y2]
        label = self.label[x1:x2, y1:y2]

        if self.flip_augmentation and self.patch_size > 1 and np.random.random() < 0.5:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.5:
                data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.5:
                data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]
        else:
            label = self.labels[i]
            
        # Add a fourth dimension for 3D CNN
        # if self.patch_size > 1:
        #     # Make 4D data ((Batch x) Planes x Channels x Width x Height)
        #     data = data.unsqueeze(0)
        # plt.imshow(data[[10,23,23],:,:].permute(1,2,0))
        # plt.show()
        return data, label

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.data, self.label = next(self.loader)

        except StopIteration:
            self.next_input = None

            return
        with torch.cuda.stream(self.stream):
            self.data = self.data.cuda(non_blocking=True)
            self.label = self.label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.data
        label = self.label

        self.preload()
        return data, label

# ULDGN: Uncertainty-Aware Language-Guided Domain Generalization Network for Cross-Scene Hyperspectral Image Classification

<p align='center'>
  <img src='PIG.png' width="800px">
</p>





## Requirements

You can create an environment by running the following code:

```
conda env create -f environment.yml
```


## Dataset

The dataset directory should look like this:

```bash
datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
└── Pavia
│   ├── paviaC.mat
│   └── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
└── HyRANK
    ├── Dioni.mat
    └── Dioni_gt_out68.mat
    ├── Loukia.mat
    └── Loukia_gt_out68.mat

```

## Usage

1.You can download [Houston; Pavia; HyRANK](https://drive.google.com/drive/folders/1No-DNDT9P1HKsM9QKKJJzat8A1ZhVmmz?usp=sharing) dataset here, and download the CLIP pre-training weight [ViT-B-32.pt](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) here.

1.You can change the `source_name` and `target_name` in train.py to set different transfer tasks.

2.Run the following command:

Houston dataset:
```
python train.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18     --lambda_1 10 --lambda_2 0.01 --lambda_3 2  --alpha 0.7  --alpha1 1 --d_se 32 --seed 30014
```
Pavia dataset:
```
python train.py --data_path ./datasets/Pavia/ --source_name paviaU --target_name paviaC   --patch_size 8 --lambda_1 10 --lambda_2 1 --lambda_3 1  --alpha 0.1  --alpha1 1 --d_se 64 --idleness_epoch 2 --seed 217
```
HyRANK dataset:
```
python train.py --data_path ./datasets/HyRank/ --source_name Dioni --target_name Loukia  --patch_size 8 --lambda_1 10 --lambda_2 1 --lambda_3 1   --alpha 0.1  --alpha1 1 --d_se 64 --idleness_epoch 2 --seed 217
```



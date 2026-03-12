import os
import os.path as osp
import numpy as np
import torch
import torchvision.transforms
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu
import matplotlib.patches as mpatches
from PIL import Image
import random
import torchvision
from pathlib import Path

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import albumentations

CLASSES = ['blue algae', 'bushfire', 'debris flow', 'farmland fire','flood',
           'forest fire', 'green tide', 'red tide', 'volcanic eruption', 'normal']

ORIGIN_IMG_SIZE = (256, 256)
INPUT_IMG_SIZE = (256, 256)

def get_test_transform():
    test_transform = [
        albu.Resize(INPUT_IMG_SIZE[0], INPUT_IMG_SIZE[1]),
        albu.Normalize()    #默认使用 ImageNet 数据集的通道均值 (0.485,0.456,0.406) 与标准差 (0.229,0.224,0.225)
    ]
    return albu.Compose(test_transform)


class MSESADDataset(Dataset):
    def __init__(self, data_root='D:/airs/ESA_CLIP/MS_ESAD_256',
                mode='train',
                transform=False):
        self.data_root = data_root
        self.transform = transform
        self.mode = mode
        self.img_ids_and_labels = open(os.path.join(self.data_root, self.mode + '.txt')).readlines()

    def __getitem__(self, index):
        img_rgb, img_swir, label, img_id, cls = self.load_img_and_label(index)
        if self.transform:
            img_rgb, img_swir = np.array(img_rgb), np.array(img_swir)
            aug1, aug2 = get_test_transform()(image=img_rgb.copy()), get_test_transform()(image=img_swir.copy())
            img_rgb, img_swir = aug1['image'], aug2['image']
        else:
            img_rgb, img_swir = np.array(img_rgb, dtype=np.uint8) / 255.0, np.array(img_swir, dtype=np.uint8) / 255.0

        img_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        img_swir = torch.from_numpy(img_swir).permute(2, 0, 1).float()

        result = dict(img_id=img_id, img_rgb=img_rgb, img_swir=img_swir, label=label, cls=cls)
        return result

    def __len__(self):
        return len(self.img_ids_and_labels)

    def load_img_and_label(self, index):
        img_info = self.img_ids_and_labels[index]
        img_id = img_info.strip().split('  ')[0]
        label = img_info.strip().split('  ')[1]
        img_name_rgb = osp.join(self.data_root, str(label), 'RGB', img_id)
        img_name_swir = osp.join(self.data_root, str(label), 'SWIR', img_id)
        img_rgb = Image.open(img_name_rgb).convert('RGB')
        img_swir = Image.open(img_name_swir).convert('RGB')
        if label in CLASSES:
            cls = CLASSES.index(label)
        else:
            cls = 255

        return img_rgb, img_swir, label, img_id, cls

def split_names(file_list, train_ratio=0.5, val_ratio=0, seed=42):
    """
    将 file_list 随机拆分为 train/val/test 三部分
    train_ratio + val_ratio <= 1.0，其余作为 test。
    """
    random.seed(seed)  # 保证可复现 :contentReference[oaicite:0]{index=0}
    n_total = len(file_list)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # 随机抽取训练集
    train_samples = random.sample(file_list, k=n_train)
    remaining = [f for f in file_list if f not in train_samples]

    # 从剩余中抽取验证集
    val_samples = random.sample(remaining, k=n_val)
    test_samples = [f for f in remaining if f not in val_samples]

    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    data_root = 'D:/airs/ESA_CLIP/MS_ESAD_256'
    classes = [
        ('blue algae', 'blue algae'),
        ('bushfire', 'bushfire'),
        ('debris flow', 'debris flow'),
        ('farmland fire', 'farmland fire'),
        ('flood', 'flood'),
        ('forest fire', 'forest fire'),
        ('green tide', 'green tide'),
        ('red tide', 'red tide'),
        ('volcanic eruption', 'volcanic eruption'),
        ('normal', 'normal'),
    ]

    # 三个全局列表用来写文件
    train_all, val_all, test_all = [], [], []

    for class_key, class_name in classes:
        rgb_dir = osp.join(data_root, class_key, 'RGB')
        file_list = os.listdir(rgb_dir)

        train_n, val_n, test_n = split_names(file_list)

        # 将 (文件名, 类别名) 加入到三个列表
        train_all += [(fn, class_name) for fn in train_n]
        val_all += [(fn, class_name) for fn in val_n]
        test_all += [(fn, class_name) for fn in test_n]


    # 将结果写入对应的 txt
    def write_list(fn, data_list):
        with open(fn, 'w') as f:
            for file_id, cls in data_list:
                f.write(f"{file_id}  {cls}\n")


    write_list('D:/airs/ESA_CLIP/train.txt', train_all)
    write_list('D:/airs/ESA_CLIP/val.txt', val_all)
    write_list('D:/airs/ESA_CLIP/test.txt', test_all)




from typing import Optional
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image

class MultiSceneDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "RSMLR", batch_size: int = 32, val_split: float = 0.2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        # 加载训练数据集，并进行拆分以创建验证集
        trainval_dataset = MultiSceneDataset(data_dir=self.data_dir, train=True)
        val_size = int(len(trainval_dataset) * self.val_split)
        train_size = len(trainval_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(trainval_dataset, [train_size, val_size],
                                                            generator=torch.Generator().manual_seed(42))

        #print(self.train_dataset)
        #exit()
        # 加载测试集
        self.test_dataset = MultiSceneDataset(data_dir=self.data_dir, train=False)
        self.num_classes = trainval_dataset.num_classes
        #print(self.num_classes)
        self.classes = trainval_dataset.classes
        #print(self.classes)
        #exit()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=15, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=15)

    def teardown(self, stage: Optional[str] = None):
        # 清理操作（如果需要）
        pass


class MultiSceneDataset(Dataset):
    def __init__(self, data_dir='RSMLR', train=True, aug_prob=0.5, img_mean=(0.485, 0.456, 0.406),
                 img_std=(0.229, 0.224, 0.225)):
        self.data_dir = data_dir
        self.train = train

        self.aug = train # 是否进行数据增强
        self.aug_prob = aug_prob

        self.img_mean = img_mean
        self.img_std = img_std

        # 根据是否为训练集加载对应的CSV文件
        csv_file = 'trainval.csv' if train else 'test.csv'
        self.data_frame = pd.read_csv(os.path.join(data_dir, csv_file))

        # 从CSV文件列名中提取类别名称（假设第一列是图像名称，其余列是类别）
        self.classes = list(self.data_frame.columns[1:])
        self.num_classes = len(self.classes)

        # Images文件夹的路径
        self.images_dir = os.path.join(data_dir, 'Images')


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.images_dir, self.data_frame.iloc[idx, 0] + '.jpg')
        # image = Image.open(img_name).convert('RGB')
        #img_name = os.path.join(self.images_dir, self.data_frame.iloc[idx, 0] + '.png')
        img_name = os.path.join(self.images_dir, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name)

        # 获取该图片的所有标签
        labels = self.data_frame.iloc[idx, 1:].values.astype('float')

        if self.aug:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=self.aug_prob),
                transforms.RandomVerticalFlip(p=self.aug_prob),
                transforms.RandomRotation(10),
                transforms.ToTensor(), # 转换为张量
                transforms.Normalize(self.img_mean, self.img_std) # 正规化
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(), # 转换为张量
                transforms.Normalize(self.img_mean, self.img_std) # 正规化
            ])

        image = self.transform(image) # 应用转换

        labels = torch.tensor(labels, dtype=torch.float)

        return image, labels
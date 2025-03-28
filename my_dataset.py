from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        #这是类的构造函数，接受images_path、images_class和transform作为参数，并将它们保存为类的属性。
        self.images_path = images_path#将传入的images_path参数保存为类的属性。
        self.images_class = images_class#将传入的images_class参数保存为类的属性。
        self.transform = transform# 将传入的transform参数保存为类的属性。

    def __len__(self):#用于返回数据集的长度。
        return len(self.images_path)#返回数据集中图像路径列表的长度。

    def __getitem__(self, item):#用于获取数据集中特定索引位置的样本
        img = Image.open(self.images_path[item])#使用PIL库中的Image.open方法打开指定索引位置的图像。
        # RGB为彩色图片，L为灰度图片
        #if img.mode != 'RGB':
        #    raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item] #获取指定索引位置的图像类别标签。

        if self.transform is not None: #检查是否有图像转换操作。
            img = self.transform(img) #如果有图像转换操作，则对图像进行相应的转换

        return img, label

    @staticmethod
    def collate_fn(batch): #用于将一个批次的样本进行整理
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch)) #将批次中的图像和标签分别提取出来。

        images = torch.stack(images, dim=0) #使用PyTorch的torch.stack方法将图像堆叠成一个张量。
        labels = torch.as_tensor(labels) #将标签转换为PyTorch张量。
        return images, labels

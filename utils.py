# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 11:36:29 2021

@author: Eric
"""
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import glob

#%%
class TrainDataset(Dataset):
    def __init__(self, img_dir, label_dir, name_list, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.names = name_list
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img_filename = os.path.join(self.img_dir, self.names[i]) + ".jpg"
        label_filename = os.path.join(self.label_dir, self.names[i]) + ".jpg"

        img = Image.open(img_filename).convert('RGB')
        img = self.transform(img)
        label = Image.open(label_filename).convert('RGB')
        label = self.transform(label)

        return (img, label)

class TestDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.transform = transform
        self.file_list = glob.glob(img_dir+"/*.jpg")
        self.names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.file_list[i]).convert('RGB')
        img = self.transform(img)

        return img, self.names[i]

def Tensor2PIL(tensor, filename=None):
    loader = transforms.ToPILImage(mode="RGB")
    img = loader(tensor.squeeze())
    # img = img.convert("RGB")
    if filename:
        img.save(filename)
    return img

def gray2rgb(tensors):
    imgs = torch.FloatTensor(tensors.size()[0], 3, tensors.size()[2], tensors.size()[3])

    for i in range(tensors.size()[0]):
        for j in range(3):
            imgs[i][j] = tensors[i]

    return imgs

def random_fliplr(imgs1, imgs2):
    outputs1 = torch.FloatTensor(imgs1.size())
    outputs2 = torch.FloatTensor(imgs2.size())
    for i in range(imgs1.size()[0]):
        if torch.rand(1)[0] < 0.5:
            img1 = torch.FloatTensor(
                (np.fliplr(imgs1[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs1.size()[1], imgs1.size()[2], imgs1.size()[3]) + 1) / 2)
            outputs1[i] = (img1 - 0.5) / 0.5
            img2 = torch.FloatTensor(
                (np.fliplr(imgs2[i].numpy().transpose(1, 2, 0)).transpose(2, 0, 1).reshape(-1, imgs2.size()[1], imgs2.size()[2], imgs2.size()[3]) + 1) / 2)
            outputs2[i] = (img2 - 0.5) / 0.5
        else:
            outputs1[i] = imgs1[i]
            outputs2[i] = imgs2[i]

    return outputs1, outputs2

def random_crop(imgs1, imgs2, crop_size = 256):
    outputs1 = torch.FloatTensor(imgs1.size()[0], imgs1.size()[1], crop_size, crop_size)
    outputs2 = torch.FloatTensor(imgs2.size()[0], imgs2.size()[1], crop_size, crop_size)
    for i in range(imgs1.size()[0]):
        img1 = imgs1[i]
        img2 = imgs2[i]
        rand1 = np.random.randint(0, imgs1.size()[2] - crop_size)
        rand2 = np.random.randint(0, imgs1.size()[3] - crop_size)
        outputs1[i] = img1[:, rand1: crop_size + rand1, rand2: crop_size + rand2]
        outputs2[i] = img2[:, rand1: crop_size + rand1, rand2: crop_size + rand2]

    return outputs1, outputs2
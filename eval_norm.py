# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:31:20 2021

@author: Eric
"""

from model import Unet
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.utils import save_image

import os
import glob
import numpy as np
from tqdm import tqdm
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
PATH_CHK = "checkpoints/norm/norm_net_epoch_020.pth"
DIR_EVAL = "test"
CROP = 1024

#%%
transform = transforms.Compose([
    transforms.Resize(CROP),
    transforms.CenterCrop(CROP),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # (input - mean) / std
    # outputs range from -1 to 1
])

class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.file_list = glob.glob(img_dir+"/*.jpg")
        self.names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.file_list[i]).convert('RGB')
        img = transform(img)

        return img, self.names[i]

#%% test
def test(net, in_folder, out_folder):
    output_normal = os.path.join(out_folder, "output")
    if not os.path.exists(output_normal):
        os.makedirs(output_normal)

    data_test = TestDataset(in_folder)
    # print(batch_size)
    testloader = DataLoader(data_test, batch_size=1, shuffle=False)

    print("\nOutput test files...")

    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            img_in = data[0].to(device)
            img_out = net(img_in)
            # print(img_name)

            img_out_filename = os.path.join(output_normal, f"{data[1][0]}.png")
            save_image(img_out, img_out_filename, value_range=(-1,1), normalize=True)

    print("Done!")

#%%
def main():
    input_folder = os.path.join(DIR_EVAL, "input")

    norm_net = Unet().to(device)
    checkpoint = torch.load(PATH_CHK)
    norm_net.load_state_dict(checkpoint["model"])

    test(norm_net, input_folder, DIR_EVAL)

if __name__ == "__main__":
    main()
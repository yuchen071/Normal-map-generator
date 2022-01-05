# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:34:39 2021

@author: Eric
"""
#%%
from model import Unet
from utils import random_fliplr, random_crop

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torchvision.utils import save_image
# from torchinfo import summary

import os
import glob
import numpy as np
from tqdm import tqdm
from time import sleep
from PIL import Image
import matplotlib.pyplot as plt

import json
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
DIR_TRAIN = "dataset/train"
DIR_VALID = "valid"
DIR_TEST = "test"
CHK_OUT = "checkpoints/norm"
TEST_CROP = 512    # px

PARAMS = {
    "Type": "Normal net",

    # "pretrain": "norm_net_epoch_200.pth",
    "pretrain": None,

    "train": {
        "epochs": 100,
        "batch": 4,
        "lr": 5e-4,
        "split": 0.9,
        "nWorkers": 2,
        },
    
    "valid": {
        "num": 2,  # should be smaller than batch size
        "log_interv": 10,
    },

    "image": {
        "img_resize": 512,
        "img_crop": 512,
        "rand_flip": True,
        "rand_crop": None
        },

    "writer": False,    # Tensorboard on/off
}

if not os.path.exists(DIR_VALID):
    os.makedirs(DIR_VALID)
if not os.path.exists(CHK_OUT):
    os.makedirs(CHK_OUT)
if PARAMS["train"]["batch"] <= PARAMS["valid"]["num"]:
    PARAMS["valid"]["num"] = PARAMS["train"]["batch"]

def pretty_json(hp):
  json_hp = json.dumps(hp, indent=2)
  return "".join("\t" + line for line in json_hp.splitlines(True))

#%%
transform = transforms.Compose([
    transforms.Resize(PARAMS["image"]["img_resize"]),
    transforms.CenterCrop(PARAMS["image"]["img_crop"]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # (input - mean) / std
    # outputs range from -1 to 1
])

test_transform = transforms.Compose([
    transforms.Resize(TEST_CROP),
    transforms.CenterCrop(TEST_CROP),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # (input - mean) / std
    # outputs range from -1 to 1
])

class TrainDataset(Dataset):
    def __init__(self, img_dir, target_dir, name_list):
        self.img_dir = img_dir
        self.target_dir = target_dir
        self.names = name_list

    def __len__(self):
        return len(self.names)

    def __str__(self):
        return self.names

    def __getitem__(self, i):
        img_filename = os.path.join(self.img_dir, self.names[i]) + ".jpg"
        target_filename = os.path.join(self.target_dir, self.names[i]) + ".jpg"

        img = Image.open(img_filename).convert('RGB')
        target = Image.open(target_filename).convert('RGB')
        img = transform(img)
        target = transform(target)

        return (img, target, self.names[i])

class TestDataset(Dataset):
    def __init__(self, img_dir):
        self.file_list = glob.glob(img_dir+"/*.jpg")
        self.names = [os.path.splitext(os.path.basename(fp))[0] for fp in self.file_list]

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = Image.open(self.file_list[i]).convert('RGB')
        img = test_transform(img)

        return img, self.names[i]

#%%
def train(img_folder, label_folder, name_list, valid_folder, pretrained=None):

    data_train = TrainDataset(img_folder, label_folder, name_list)
    num_train = int(len(data_train) * PARAMS["train"]["split"])
    data_train, data_valid = random_split(data_train, [num_train, len(data_train) - num_train])

    print("Train data: %d, Validation data: %d, Train batches: %.2f\n" %  \
          (len(data_train), len(data_valid), len(data_train)/PARAMS["train"]["batch"]))

    trainloader = DataLoader(data_train, batch_size=PARAMS["train"]["batch"],
                             num_workers=PARAMS["train"]["nWorkers"], shuffle=True, drop_last=True)
    validloader = DataLoader(data_valid, batch_size=PARAMS["train"]["batch"], shuffle=False, num_workers=2)

    net = Unet()
    net.weight_init(mean=0.0, std=0.02)
    net.to(device)
    # summary(net, (1, 3, 512, 512))

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=PARAMS["train"]["lr"], betas=(0.5, 0.999))

    # train
    train_loss_hist = []
    valid_loss_hist = []

    if pretrained:
        checkpoint = torch.load(os.path.join(CHK_OUT, pretrained))
        net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optim"])
        train_loss_hist = checkpoint["train_loss_hist"]
        valid_loss_hist = checkpoint["valid_loss_hist"]
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    # fixed valid output
    v_num = PARAMS["valid"]["num"]
    if len(data_valid) <= v_num:
        v_num = len(data_valid)    # on the off-chance valid dataset only has 1 image

    valid_img_data = next(iter(validloader))
    valid_img_data = [data[:v_num] for data in valid_img_data]
    if PARAMS["image"]["rand_crop"]:
        valid_img_data[0], valid_img_data[1] = random_crop(valid_img_data[0], valid_img_data[1], PARAMS["image"]["rand_crop"])

    # tensorboard
    if PARAMS["writer"]:
        writer = SummaryWriter()
        writer.add_text("Parameters", pretty_json(PARAMS), 0)
        writer.add_text("Validation images", str(valid_img_data[2]), 0)

    sleep(0.3)
    for epoch in range(start_epoch, PARAMS["train"]["epochs"]):
        # train
        pbar = tqdm(trainloader, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
        p_desc = "Train %2d/%d" % (epoch + 1, PARAMS["train"]["epochs"])
        pbar.set_description(p_desc)

        net.train()
        tmp_loss = []
        for batch_id, (img_in, target, _) in enumerate(pbar):

            if PARAMS["image"]["rand_flip"]:
                img_in, target = random_fliplr(img_in, target)

            if PARAMS["image"]["rand_crop"]:
                img_in, target = random_crop(img_in, target, PARAMS["image"]["rand_crop"])

            img_in = img_in.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            img_out = net(img_in)
            loss = criterion(img_out, target)
            loss.backward()
            optimizer.step()

            tmp_loss.append(loss.item())

            p_post = f"T_Loss: {loss.item(): .4f}"
            pbar.set_postfix_str(p_post)
            pbar.update(0)

        train_loss_hist.append(np.mean(tmp_loss))

        if PARAMS["writer"]:
            writer.add_scalar("Loss/Train", train_loss_hist[-1], epoch)

        # validation
        pbar = tqdm(validloader, ascii=True, bar_format='{l_bar}{bar:10}{r_bar}')
        p_desc = "Valid %2d/%d" % (epoch + 1, PARAMS["train"]["epochs"])
        pbar.set_description(p_desc)

        net.eval()
        tmp_loss = []
        with torch.no_grad():
            for batch_id, (img_in, target, _) in enumerate(pbar):

                if PARAMS["image"]["rand_flip"]:
                    img_in, target = random_fliplr(img_in, target)

                if PARAMS["image"]["rand_crop"]:
                    img_in, target = random_crop(img_in, target, PARAMS["image"]["rand_crop"])

                img_in = img_in.to(device)
                target = target.to(device)

                img_out = net(img_in)
                loss = criterion(img_out, target)

                tmp_loss.append(loss.item())

                p_post = f"V_Loss: {loss.item(): .4f}"
                pbar.set_postfix_str(p_post)
                pbar.update(0)

            valid_loss_hist.append(np.mean(tmp_loss))

            if PARAMS["writer"]:
                writer.add_scalar("Loss/Valid", valid_loss_hist[-1], epoch)

        if (epoch+1) % PARAMS["valid"]["log_interv"] == 0 or epoch == 0:
            with torch.no_grad():
                img_in = valid_img_data[0].to(device)
                target = valid_img_data[1].to(device)
                img_out = net(img_in)

                imgs = torch.cat([img_in, target, img_out])
                save_image(imgs, os.path.join(valid_folder, f"epoch_{epoch+1}.png"),
                           value_range=(-1,1), normalize=True, nrow=v_num)

    # save pth
    torch.save({
        "epoch": epoch+1,
        "model": net.state_dict(),
        "optim": optimizer.state_dict(),
        "train_loss_hist": train_loss_hist,
        "valid_loss_hist": valid_loss_hist
        }, os.path.join(CHK_OUT, f"norm_net_epoch_{epoch+1:03}.pth"))

    plotLoss(train_loss_hist, valid_loss_hist, "Loss history")

    # tensorboard
    if PARAMS["writer"]:
        writer.flush()
        writer.close()

    return net

#%% test
def test(net, in_folder, out_folder):
    data_test = TestDataset(in_folder)
    batch_size = len(data_test)
    # print(batch_size)
    testloader = DataLoader(data_test, batch_size=batch_size, shuffle=False)

    print("\nOutput test files...")

    net.eval()
    with torch.no_grad():
        for idx, data in enumerate(testloader):
            img_in = data[0].to(device)
            img_out = net(img_in)
            # print(img_name)
            out_filename = os.path.join(out_folder, "output.png")
            save_image(torch.cat([img_in, img_out]), out_filename, value_range=(-1,1), normalize=True, nrow=batch_size)

    print("Done!")

#%%
def plotLoss(t_hist, v_hist, title):
    plt.figure()
    plt.plot(t_hist, label="Train")
    plt.plot(v_hist, label="Valid")
    plt.title(title)
    plt.legend()
    plt.xlabel("Epochs")
    plt.show()

#%%
def main():
    # ==== train normal ====
    print("Normal map")
    color_folder = os.path.join(DIR_TRAIN, "color")
    norm_folder = os.path.join(DIR_TRAIN, "normal")
    name_txt = os.path.join(DIR_TRAIN, "name_list.txt")

    with open(name_txt, "r") as f:
        name_list = [line.rstrip('\n') for line in f.readlines()]

    test_in_folder = os.path.join(DIR_TEST, "input")
    test_norm_folder = os.path.join(DIR_TEST, "output_norm")
    valid_folder = os.path.join(DIR_VALID, "norm")
    if not os.path.exists(test_norm_folder):
        os.makedirs(test_norm_folder)
    if not os.path.exists(valid_folder):
        os.makedirs(valid_folder)

    norm_net = train(color_folder, norm_folder, name_list, valid_folder, pretrained=PARAMS["pretrain"])
    test(norm_net, test_in_folder, test_norm_folder)
    if str(device) == 'cuda':
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()


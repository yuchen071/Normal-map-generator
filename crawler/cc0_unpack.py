# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:01:56 2021

@author: Eric
"""

import os
import io
from PIL import Image
from tqdm import tqdm
from zipfile import ZipFile, BadZipFile

DIR_IN = "../dataset/zip"
DIR_OUT = "../dataset/train"   # + color, normal, displace

color_dir = os.path.join(DIR_OUT, "color")
normal_dir = os.path.join(DIR_OUT, "normal")
displace_dir = os.path.join(DIR_OUT, "displacement")

if not os.path.exists(DIR_OUT):
    os.makedirs(DIR_OUT)
if not os.path.exists(color_dir):
    os.makedirs(color_dir)
if not os.path.exists(normal_dir):
    os.makedirs(normal_dir)
if not os.path.exists(displace_dir):
    os.makedirs(displace_dir)

#%%
def main():
    zip_list = [name for name in os.listdir(DIR_IN) if ".zip" in name]
    name_list = [name.split("_")[0] for name in zip_list]

    for zip_name in tqdm(zip_list):
        filename = zip_name.split("_")[0]
        try:
            with ZipFile(os.path.join(DIR_IN, zip_name), "r") as z:

                checklist = [0, 0, 0]
                for file in z.namelist():
                    if "Color" in file:
                        checklist[0] = 1
                        color_file = file

                    if "NormalGL" in file:
                        checklist[1] = 1
                        norm_file = file

                    if "Displacement" in file:
                        checklist[2] = 1
                        disp_file = file

                if 0 not in checklist:
                    z.extract(color_file, color_dir)
                    os.replace(os.path.join(color_dir, color_file), os.path.join(color_dir, filename) + ".jpg")

                    z.extract(norm_file, normal_dir)
                    os.replace(os.path.join(normal_dir, norm_file), os.path.join(normal_dir, filename) + ".jpg")

                    z.extract(disp_file, displace_dir)
                    os.replace(os.path.join(displace_dir, disp_file), os.path.join(displace_dir, filename) + ".jpg")

        except BadZipFile:
            pass

    print("\nExtraction Done!")

    name_list = sorted(name_list)
    with open(os.path.join(DIR_OUT, "name_list.txt"), "w") as output:
        for name in name_list:
            output.write(name + "\n")

if __name__ == "__main__":
    main()
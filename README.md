# Normal map generator
Aside from Base Color texture images, 3D texture packs often include images such as Normal maps, Displacement maps to make the texture look more realistic. Blender Guru has an excellent video explaining what each texture map does in a 3D animation software: [Link](https://www.blenderguru.com/tutorials/basics-realistic-texturing) 

![Textureing](https://images.squarespace-cdn.com/content/v1/58586fa5ebbd1a60e7d76d3e/1494407035745-K9TK8VNTL62E3J98RCU5/image-asset.jpeg?format=400w)

A texture pack from [ambientcg.com](https://ambientcg.com/) includes images below:  
| Base Color | Normal Map | Displacement Map | Roughness |
|:--:|:--:|:--:|:--:|
| ![color](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/brick_texture/color.jpg) | ![normal](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/brick_texture/normal.jpg) | ![disp](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/brick_texture/displacement.jpg) | ![rough](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/brick_texture/roughness.jpg) |

This project aims to generate normal maps and displacement maps automatically with UNet, using texture images from [ambientcg.com](https://ambientcg.com/) as our training dataset.

## Results
Training output:  
![out1](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/results/output1.png)  
![out2](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/results/output2.png)

Blender demo with custom photo texture:
| No Normal Map | With Normal Map |
|:--:|:--:|
| ![nonorm](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/results/nonormal.gif) | ![withnorm](https://github.com/yuchen071/Normal-map-generator/blob/main/.readme_docs/results/withnormal.gif) |

## Requirements
The root folder should be structured as follows:
```
📁 root/
  ├─ 📁 crawler/
  |  ├─ 📄 cc0_crawler.py
  |  └─ 📄 cc0_unpack.py
  |
  ├─ 📁 test/
  |  └─ 📁 input/
  |     ├─ 🎨 image1.jpg
  |     ├─ 🎨 image2.jpg
  |     └─ ...
  |
  ├─ 📄 eval_disp.py
  ├─ 📄 eval_norm.py
  ├─ 📄 model.py
  ├─ 📄 train_disp.py
  ├─ 📄 train_norm.py
  └─ 📄 utils.py
```

The `test/input` folder should contain your own images you wish to be converted.

### Dependencies
```
torchinfo==0.1.1
matplotlib==3.3.4
numpy==1.19.2
torchvision==0.9.0
fake_useragent==0.1.11
torch==1.8.0
requests==2.26.0
tqdm==4.62.2
beautifulsoup4==4.10.0
Pillow==8.4.0
```

## How to use
### Web Crawler
Run the following scripts from the crawler folder in order:
```
python cc0_crawler.py
python cc0_unpack.py
```
A `dataset` folder should appear in the root folder containing texture packs and unzipped images downloaded from ambientcg.com:
```
📁 root/
  ├─ 📁 crawler/
  |  ├─ 📄 cc0_crawler.py
  |  └─ 📄 cc0_unpack.py
  |
  ├─ 📁 dataset/
  |  ├─ 📁 train/
  |  |  ├─ 📁 color/
  |  |  |  ├─ 🎨 texture001.jpg
  |  |  |  ├─ 🎨 texture002.jpg
  |  |  |  └─ 🎨 ...
  |  |  |
  |  |  ├─ 📁 displacement/
  |  |  |  ├─ 🎨 texture001.jpg
  |  |  |  ├─ 🎨 texture002.jpg
  |  |  |  └─ 🎨 ...
  |  |  |
  |  |  ├─ 📁 normal/
  |  |  |  ├─ 🎨 texture001.jpg
  |  |  |  ├─ 🎨 texture002.jpg
  |  |  |  └─ 🎨 ...
  |  |  |
  |  |  └─ 📝 name_list.txt
  |  |
  |  └─📁 zip/
  |     ├─ 📚 texture001.zip
  |     ├─ 📚 texture002.zip
  |     └─ 📚 ...
  ...
```
### Train & Test
Run the following script to train a network that generates normal maps:
```
python train_norm.py
```
Run the following script to train a network that generates displacement maps:
```
python train_disp.py
```
A `valid` folder should appear in root during training containing validation sample outputs.  
A `checkpoints` folder should also appear after training script ends containing trained network `.pth` files.

### Evaluation




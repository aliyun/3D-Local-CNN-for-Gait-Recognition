# 3D Local Convolutional Neural Networks for Gait Recognition

### Overview
This repository contains the training code of 3DLocalCNN introduced
in our paper.

In this work, we present a new building block for 3D CNNs with local
information incorporated, termed as 3D local convolutional neural networks. Our
local operations can be combined with any existing architectures. We
demonstrate the superiority of local operations on the task of gait recognition
where 3D local CNN consistently outperforms state-of-the-art models. We hope
this work will shed light on more research on introducing simple but effective
local operations as submodules of existing convolutional building blocks.

### Installation Instructions
- Clone this repo:

```bash
git clone git@github.com:aliyun/3D-Local-Convolutional-Neural-Networks-for-Gait-Recognition.git
cd 3D-Local-Convolutional-Neural-Networks-for-Gait-Recognition
```

- Create a conda virtual environment and activate it:

```bash
conda create -n 3DLocalCNN python=3.6.4 -y
conda activate 3DLocalCNN
```

- Install `CUDA==10.2` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.2`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
```

- Install `numpy==1.16.4, yaml, tensorboard, pyyaml, scikit-learn, opencv-python, imageio, matplotlib, seaborn, xarray`:

```bash
pip3 install numpy==1.16.4, yaml, tensorboard, pyyaml, scikit-learn, opencv-python, imageio, matplotlib, seaborn, xarray
```

### Usage
#### Data preprocess

 Download CASAI raw data to data/CASIA_raw and run `python preprocess.py`

#### Demo
(1) To train GaitSet from scratch, run
```
python main.py --config=configs/GaitSet_CASIA.yaml
```
(2) To train GaitPart from scratch, run
```
python main.py --config=configs/GaitPart_CASIA.yaml
```
(3) To train 3DLocalCNN from scratch, run
```
python main.py --config=configs/3DLocalCNN_CASIA.yaml
```

### License
+ Apache License 2.0


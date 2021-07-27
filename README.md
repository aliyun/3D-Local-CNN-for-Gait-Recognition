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

### Run environment

+ Python 3.7
+ Python bindings for OpenCV
+ Pytorch 1.1

### Usage
#### Data preprocess

 Download CASAI raw data to data/CASIA_raw and run `python preprocess.py`

#### Demo
```
(1) basic run, for all gpus
```
python main.py --config=configs/3DLocalCNN_CASIA.yaml
```
(2) for 1 gpu, add `CUDA_VISIBLE_DEVICES=1`. e.g.
```
CUDA_VISIBLE_DEVICES=1 python main.py --config=configs/3DLocalCNN_CASIA.yaml
```
(3) for deterministic set seed, e.g. 1234
```
python main.py --seed=1234 --config=configs/3DLocalCNN_CASIA.yaml
```

(4) add name, save log dir as 'name_time', e.g.
```
python main.py --seed=1234 --config=configs/3DLocalCNN_CASIA.yaml --name=sgd
```

(end) for nohup run, see  nohup_run.sh.example


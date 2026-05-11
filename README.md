# [ICML2026] Rethinking Loss Reweighting for Imbalance Learning as an Inverse Problem: A Neural Collapse Point of View

## Overview

This repository implements **Inverse Loss Reweighting** for long-tailed classification.

In long-tailed learning, head classes dominate the training process because they contain many more samples than tail classes. Standard cross-entropy therefore tends to produce biased classifiers that perform well on head classes but poorly on tail classes.

Our proposed reweighting strategy addresses this problem by rethinking loss reweighting from a **Neural Collapse** perspective. Under the ideal Neural Collapse geometry, all classes should have equal average loss. Based on this observation, our method treats equal per-class average loss as the target of reweighting, and formulates reweighting as an **inverse problem**. Given the current class-wise losses, it dynamically solves for the class weights that make the effective class losses closer to the ideal equal-loss target.



## Environment Setup
```bash
conda create -n ILR python=3.10 -y
conda activate ILR

pip install torch torchvision torchaudio
pip install numpy matplotlib pillow scikit-learn tensorboardX
```
If you are using a CUDA machine, install the PyTorch version that matches your CUDA driver. See the official PyTorch installation command for your platform.

## Dataset Preparation
Download the datasets CIFAR-10, CIFAR-100, ImageNet, and iNaturalist18 to /data. 
When running `train.py`, CIFAR-10 or CIFAR-100 will be downloaded automatically to `/data` if it is not already available.
If your dataset path is different, modify the following lines in `train.py`:

```python
train_set = IMBALANCECIFAR10('/your/data/path', ...)
val_set = datasets.CIFAR10('/your/data/path', ...)

train_set = IMBALANCECIFAR100('/your/data/path', ...)
val_set = datasets.CIFAR100('/your/data/path', ...)
```

## Quick Start

### Train on CIFAR-100-LT

````
python -m train --dataset cifar100 --imb_factor 0.01 --lr 0.35 --alpha 0 --switch_epoch 160 --macro_gamma 1.0 --macro_switch_epoch 160
````

### Train on CIFAR-10-LT
````
python -m train --dataset cifar10 --imb_factor 0.01 --lr 0.3 --alpha 0 --switch_epoch 160 --macro_gamma 1.0 --macro_switch_epoch 160
````
More hyper-parameter details are shown in Appendix F in our paper.



## Acknowledgements
This long-tailed classification code is based on the repository [GBG_v1](https://github.com/WickyLee1998/GBG_v1).
 

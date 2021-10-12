# Blind Image Denoising
Implementing CVPR 2020 paper : 

"ROBUST AND INTERPRETABLE BLIND IMAGE DENOISING VIA BIAS - FREE CONVOLUTIONAL NEURAL NETWORKS"

## Target
The target is to create an explainable bias-free 
model that performs denoising on a three channel input image.

## Corruption types
In order to train such a model we corrupt an input image using 
several type of noise and then try to recover the original image

* normally distributed additive noise
* normally distributed multiplicative noise
* subsampling

## Model types
* resnet
* resnet with sparse constraint
* resnet with on/off gates 
* all of the above models with multiscale processing

## Training
1. prepare training input
2. prepare training configuration
3. run training
4. export

## Image examples
TODO

## How to use
TODO



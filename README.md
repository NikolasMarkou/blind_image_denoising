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

## Training configuration
The training configuration is in the form of a json file that follows the schema:
```json
{
  "model": {
    ...
  },
  "train": {
    ...
    "optimizer": {
      ...
    }
  },
  "loss": {
    ...
  },
  "dataset": {
    ...
  }
}

```
### Model

#### Example
```json
  "model": {
    "levels": 4,
    "filters": 16,
    "no_layers": 4,
    "min_value": 0,
    "max_value": 255,
    "kernel_size": 3,
    "type": "resnet",
    "batchnorm": true,
    "stop_grads": false,
    "activation": "relu",
    "output_multiplier": 1.0,
    "kernel_regularizer": "l1",
    "final_activation": "tanh",
    "input_shape": ["?", "?", 3],
    "kernel_initializer": "glorot_normal"
  }
```
### Train
#### Example
```json
  "train": {
    "epochs": 20,
    "total_steps": -1,
    "iterations_choice": [1],
    "checkpoints_to_keep": 3,
    "checkpoint_every": 10000,
    "visualization_number": 5,
    "visualization_every": 100,
    "optimizer": {
      "decay_rate": 0.9,
      "decay_steps": 50000,
      "learning_rate": 0.001,
      "gradient_clipping_by_norm": 1.0
    }
  }
```
### Loss
#### Example
```json
  "loss": {
    "hinge": 2.5,
    "mae_multiplier": 1.0,
    "regularization": 0.01
  }
```
### Dataset
#### Example
```json
  "dataset": {
    "batch_size": 16,
    "min_value": 0,
    "max_value": 255,
    "clip_value": true,
    "random_blur": true,
    "subsample_size": -1,
    "random_invert": false,
    "random_rotate": 0.314,
    "random_up_down": true,
    "random_left_right": true,
    "dataset_shape": [256, 768],
    "input_shape": [256, 256, 3],
    "additional_noise": [5, 10, 20],
    "multiplicative_noise": [0.1, 0.15, 0.2],
    "directory": "/media/data1_4tb/datasets/KITTI/data/depth/raw_image_values/"
  }
```


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
* all the above models with multiscale processing

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
    
  },
  "train": {
    
    "optimizer": {
      
    }
  },
  "loss": {
    
  },
  "dataset": {
    
  }
}

```
### model
Describes the type and characteristics of model.
* `levels`: how many multiscale models to build.
* `filters`: the number of filters per conv operations
* `no_layers`: number of layers per level
* `min_value`: the minimum value 
* `max_value`: the maximum value
* `kernel_size`: the [NxN] filter size of each convolution filter
* `type`: type of mode `(resnet, sparse_resnet, gatenet)`
* `batchnorm`: use batch normalization between layers
* `stop_grads`: if true stop gradients from flowing to upper levels
* `activation`: convolution activation
* `output_multiplier`: multiply output with this value to avoid saturation before going to `final_activation`
* `kernel_regularizer`: kernel regularization `(l1, l2, l1_l2)` 
* `final_activation`: final activation at the end of the model
* `input_shape`: the input shape (minus the batch)
* `kernel_initializer`: kernel initializer 

#### example
```json
{
  "levels": 5,
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
### train
Describes how to train the model above.
* `epochs`: how many epoch to run the training
* `total_steps`: how many steps to run the training (set negative to ignore)
* `iterations_choice: how many times to run the model on the same input (if not sure set to `[1]`)
* `checkpoints_to_keep`: how many checkpoints to keep
* `checkpoint_every`: how many iterations before a checkpoint 
* `visualization_number`: how many visualizations to show
* `visualization_every`: show visualizations every this many iterations
* `random_batch_iterations`:
* `random_batch_size`:
* `random_batch_min_difference`:
* `optimizer`:
  * `decay_rate`:
  * `decay_steps`:
  * `learning_rate`:
  * `gradient_clipping_by_norm`:
  
#### example
```json
{
  "epochs": 20,
  "total_steps": -1,
  "iterations_choice": [1],
  "checkpoints_to_keep": 3,
  "checkpoint_every": 10000,
  "visualization_number": 5,
  "visualization_every": 100,
  "random_batch_iterations": 20,
  "random_batch_size": [512, 512, 3],
  "random_batch_min_difference": 0.01,
  "optimizer": {
    "decay_rate": 0.9,
    "decay_steps": 50000,
    "learning_rate": 0.001,
    "gradient_clipping_by_norm": 1.0
  }
 }
```
### loss
Describes how the loss function is composed.
* `hinge`:
* `mae_multiplier`:
* `regularization`:

#### example
```json
{
  "hinge": 2.5,
  "mae_multiplier": 1.0,
  "regularization": 0.01
}
```
### dataset
Describes how the dataset is parsed and prepared.
* `batch_size`: batch size for each iteration
* `min_value`: minimum value (usually 0) 
* `max_value`: maximum value (usually 255)
* `clip_value`: if true clip values at the end to minimum, maximum
* `random_blur`: if true apply blur in multiplicative noise
* `subsample_size`: if greater than zero enable subsampling noise 
* `random_invert`: if true randomly invert result
* `random_rotate`: maximum radians for rotation augmentation
* `random_up_down`: if true randomly invert up down
* `random_left_right`: if true randomly invert left right
* `dataset_shape`: resize input dataset to this size (height, width)
* `input_shape`: randomly crop this size from each input image
* `additional_noise`: select randomly additive noise with mean 0 and std deviation from this list
* `multiplicative_noise`: select randomly multiplicative noise with mean 1 and std deviation from this list
* `directory`: path to the images directory

#### example
```json
  {
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


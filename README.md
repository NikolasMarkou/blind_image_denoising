<div align="center">

**A library for blind image denoising algorithms using bias free denoising CNN's**
___

[Getting Started](#Pretrained) •
[License](https://github.com/NikolasMarkou/blind_image_denoising/blob/main/LICENSE)

[![python](https://img.shields.io/badge/python-3.6%2B-green)]()
[![tensorflow](https://img.shields.io/badge/tensorflow-2.6%2B-green)]()

</div>

___

# Blind Image Denoising

## Target
The target is to create a series of:

* interpretable
* multi scale
* high performance
* low memory footprint
 
models that performs denoising on an input (grayscale or colored) image. 

The bias-free nature of the model allows for easy interpretation and use as prior
for hard inverse problems.

## Interpretation 
Interpretation comes naturally by implementing the CVPR 2020 paper : 

["ROBUST AND INTERPRETABLE BLIND IMAGE DENOISING VIA BIAS - FREE CONVOLUTIONAL NEURAL NETWORKS"](https://arxiv.org/abs/1906.05478)

This paper provides excellent results

![](images/readme/bfcnn_noisy_1.png "single channel bias free denoising")

which can also be completely interpretable as a mask per pixel

![](images/readme/bfcnn_noisy_2.png "pixel smoothing interpretability")


## Corruption types
In order to train such a model we corrupt an input image using 
several types of noise and then try to recover the original image

* subsampling noise
* normally distributed additive noise (same per channel / different same per channel)
* normally distributed multiplicative noise (same per channel / different same per channel)

## Image examples

The following samples are `256x256` crops from the `KITTI` dataset, 
denoised using the `resnet_color_laplacian_3x5_non_shared_bn_16x3x3_128x128_residual` model.

We add truncated normal noise with different standard deviations and calculate the 
`Mean Absolute Error (MAE)` both for the noisy images, and the denoised images. 
The pixel range is 0-255. 

We can clearly see that the model adapts well to different ranges of noise.

noise (std) | MAE (noisy) | MAE (denoised) | Normal - Noisy - Denoised                     |
---|-----|-----|----------------------------------------------|
1  |0.65|4.33 |![](images/readme/bfcnn_image_0.png "normal") |
5  |3.50|3.39 |![](images/readme/bfcnn_image_1.png "normal") |
10 |6.44|5.19 |![](images/readme/bfcnn_image_2.png "normal") |
20 |13.22|6.60 |![](images/readme/bfcnn_image_3.png "normal") |
30 |19.84|8.46 |![](images/readme/bfcnn_image_4.png "normal") |
40 |27.02|12.95|![](images/readme/bfcnn_image_5.png "normal") |
50 |30.59|15.06|![](images/readme/bfcnn_image_6.png "normal") |
60 |34.34|17.81|![](images/readme/bfcnn_image_7.png "normal") |
70 |40.64|22.36|![](images/readme/bfcnn_image_8.png "normal") |
80 |45.68|27.99|![](images/readme/bfcnn_image_9.png "normal") |


## How to use (from scratch)

1. prepare training input
2. prepare training configuration
3. run training
4. export to tflite and saved_model format
5. use models

### Train
Prepare a training configuration and train with the following command:  
```bash
python -m bfcnn.train \ 
  --model-directory ${TRAINING_DIR} \ 
  --pipeline-config ${PIPELINE}
```
### Export
Export to frozen graph and/or tflite with the following command:
```bash
python -m bfcnn.export \
    --checkpoint-directory ${TRAINING_DIR} \
    --pipeline-config ${PIPELINE} \
    --output-directory ${OUTPUT_DIR} \
    --to-tflite
```

## How to use (pretrained)

Use any of the pretrained models included in the package:
* [resnet_color_1x5_non_shared_bn_16x3x3_128x128](bfcnn/pretrained/resnet_color_1x5_non_shared_bn_16x3x3_128x128)
* [resnet_color_1x5_non_shared_bn_16x3x3_128x128_skip_input](bfcnn/pretrained/resnet_color_1x5_non_shared_bn_16x3x3_128x128_skip_input)
* [resnet_color_laplacian_2x5_non_shared_bn_16x3x3_128x128_skip_input](bfcnn/pretrained/resnet_color_laplacian_2x5_non_shared_bn_16x3x3_128x128_skip_input)

```python
import bfcnn
import tensorflow as tf

# load model
denoiser_model = \
    bfcnn.load_model(
        "resnet_color_1x5_non_shared_bn_16x3x3_128x128")

# create random tensor
input_tensor = \
    tf.random.uniform(
        shape=[1, 256, 256, 3],
        minval=0,
        maxval=255,
        dtype=tf.int32)
input_tensor = \
    tf.cast(
        input_tensor,
        dtype=tf.uint8)

# run inference
denoised_tensor = denoiser_model(input_tensor)
```

## Model types
We have used traditional (bias free) architectures.
* resnet
* resnet with sparse constraint
* resnet with on/off per resnet block gates 
* all the above models with multi-scale processing

## Multi-Scale
The system is trained in multiple scales by implementing ideas 
from LapSRN (Laplacian Pyramid Super-Resolution Network) and MS-LapSRN (Multi-Scale Laplacian Pyramid Super-Resolution Network)

![](images/readme/laplacian_pyramid_decomposition.png "laplacian pyramid decomposition")

![](images/readme/laplacian_pyramid_recomposition.png "laplacian pyramid recomposition")

## Low-Memory footprint
By using a gaussian pyramid and a shared bias-free CNN model between each scale, 
we can ensure that we have a small enough model to run on very small devices while ensure we
have a big enough ERF (effective receptive field) for the task at hand.


### Additions
#### [Multi-Scale Laplacian Pyramid](bfcnn/pyramid.py)
Our addition (not in the paper) is the laplacian multi-scale pyramid
that expands the effective receptive field without the need to add many more layers (keeping it cheap computationally).
![](images/readme/laplacian_model.png "Laplacian model")

Which breaks down the original image into 3 different scales and processes them independently:

![](images/readme/laplacian_decomposition_lena.png "Laplacian Decomposition Lena")

We also have the option to add residuals at the end of each processing levels, so it works like an iterative process:

![](images/readme/laplacian_model_residual.png "Laplacian model residual")


#### [Multi-Scale Gaussian Pyramid](bfcnn/pyramid.py)
Our addition (not in the paper) is the gaussian multi-scale pyramid
that expands the effective receptive field without the need to add many more layers (keeping it cheap computationally).

![](images/readme/gaussian_decomposition_lena.png "Gaussian Decomposition Lena")

#### [Noise estimation model Multi-Scale mixer](bfcnn/model_noise_estimation.py)
Our addition (not in the paper) is a noise estimation model that
decides the contribution of each layer when mixing them back in.

#### [Squeeze and Excite with residual](bfcnn/utilities.py)
Every resnet block has the option to include a residual squeeze and excite element (not in the paper) to it.

#### Normalization layer
Our addition (not in the paper) is a (non-channel wise and non-learnable) normalization layer (not BatchNorm) 
after the DepthWise operations. This is to enforce sparsity with the differentiable relu below.

#### Differentiable RELU
Our addition (not in the paper) is a differentiable relu for specific operations.
![](images/readme/differentiable_relu.png "Differentiable RELU")

#### [Soft Orthonormal Regularization](bfcnn/regularizer.py)
Added optional orthogonality regularization constraint as found in paper [Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?](https://arxiv.org/abs/1810.09102).
This forces a soft ortho-normal constraint on the kernels.

#### [Soft Orthogonal Regularization](bfcnn/regularizer.py)
Custom regularization that forces a soft orthogonal constraint on the kernels while still allowing the kernels to grow independently or shrink to almost zero.


## References
1. [Robust and interpretable blind image denoising via bias-free convolutional neural networks](https://arxiv.org/abs/1906.05478)
2. [Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks](https://arxiv.org/abs/1710.01992)
3. [Densely Residual Laplacian Super-Resolution](https://arxiv.org/abs/1906.12021)
4. [Can We Gain More from Orthogonality Regularizations in Training Deep CNNs?](https://arxiv.org/abs/1810.09102)
5. [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)





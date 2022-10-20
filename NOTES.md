# Technical Notes

## Performance

### resnet_color_1x6_bn_32x64x32_1x3x1_256x256_channelwise_erf_relu

architecture: resnet

* depth: 1x6
* filters: 32x64x32
* kernels: 1x3x1
* resolution: 256x256
* extra: 
  * channelwise -> with 0.001 turns off around 16 feature maps completely
  * erf: l1 0.025
  * relu
  * batchnorm
* parameters: 135k

results
 
 * mae: 3.5
 * snr: 7.1db

#### Notes
indication for higher channelwise regularization

## General

* Channelwise multiplier works well but regularization needs to be an order higher than the conv layers
* Multiplier gives slighlty better results if combined with Channelwise but regularization needs to be an order higher than Channelwise
* SoftOrthogonal works but we need to find correct parameters and it is very slow to train
* Erf works well by giving incentive to convolutions to spread out

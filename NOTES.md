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

### resnet_color_1x6_bn_32x64x32_1x3x1_256x256_channelwise_so_erf_relu

architecture: resnet

* depth: 1x6
* filters: 32x64x32
* kernels: 1x3x1
* resolution: 256x256
* extra: 
  * channelwise 
  * erf: l1 0.025
  * so: 0.1
  * relu
  * batchnorm
* parameters: 135k

results
 
 * mae: 3.5
 * snr: 7.1db

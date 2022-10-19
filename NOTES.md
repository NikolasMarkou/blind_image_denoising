# Technical Notes

## Performance

resnet

* depth: 1x6
* filters: 32x64x32
* kernels: 1x3x1
* resolution: 256x256
* extra: 
  * channelwise -> turns off around 16 feature maps completely
  * erf
  * relu
  * batchnorm
* parameters: 135k

results
 
 * mae: 3.5
 * snr: 7.1db

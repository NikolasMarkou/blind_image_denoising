PYRAMID and ERF
```json
"model_denoise": {
    "filters": 32,
    "no_layers": 6,
    "kernel_size": 7,
    "block_kernels": [3, 3],
    "block_filters": [32, 32],
    "value_range": [0, 255],
    "type": "resnet",
    "batchnorm": true,
    "activation": "relu6",
    "final_activation": "tanh",
    "input_shape": ["?", "?", 3],
    "kernel_initializer": "glorot_normal",
    "kernel_regularizer": {
        "type": "erf",
        "config": {
            "l2_coefficient": 0.0,
            "l1_coefficient": 0.025
        }
    },
    "pyramid": {
      "levels": 3,
      "type": "laplacian",
      "kernel_size": [5, 5]
    }
  }
```

MULTISCALE and COUNT NON-ZERO MEAN
```
  "loss": {
    "hinge": 0.5,
    "cutoff": 255.0,
    "mae_multiplier": 1.0,
    "regularization": 0.01,
    "use_multiscale": true,
    "count_non_zero_mean": true
  }
```

SELECTOR
```json
  "model_denoise": {
    "filters": 32,
    "no_layers": 6,
    "kernel_size": 7,
    "block_kernels": [3, 3],
    "block_filters": [32, 32],
    "value_range": [0, 255],
    "type": "resnet",
    "batchnorm": true,
    "activation": "relu",
    "final_activation": "tanh",
    "add_selector": true,
    "input_shape": ["?", "?", 3],
    "kernel_initializer": "glorot_normal",
    "kernel_regularizer": "l1"
  }
```

PRUNING
```json
   "train": {
    "epochs": 20,
    "total_steps": -1,
    "use_test_images": true,
    "checkpoints_to_keep": 3,
    "checkpoint_every": 10000,
    "visualization_number": 4,
    "visualization_every": 1000,
    "same_sample_iterations": 1,
    "random_batch_iterations": 5,
    "random_batch_size": [256, 256, 3],
    "random_batch_min_difference": 0.1,
    "prune": {
      "start_epoch": 0,
      "steps": 1000,
      "strategies": [{
        "type": "minimum_threshold",
        "config": {
          "minimum_threshold": 0.001
        }
      }]
    }
```

SOFT-ORTHONORMAL
```json
    "uncertainty_kernel_regularizer": {
        "type": "soft_orthonormal",
        "config": {
            "lambda_coefficient": 0.1,
            "l1_coefficient": 0.001
      }
    }
```
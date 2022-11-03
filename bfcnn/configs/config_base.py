config = {
  "model_denoise": {

  },
  "train": {
    "epochs": 20,
    "total_steps": -1,
    "use_test_images": True,
    "checkpoints_to_keep": 3,
    "checkpoint_every": 10000,
    "visualization_number": 4,
    "visualization_every": 1000,
    "same_sample_iterations": 1,
    "random_batch_iterations": 5,
    "random_batch_size": [256, 256, 3],
    "random_batch_min_difference": 0.1,
    "optimizer": {
      "gradient_clipping_by_norm": 1.0,
      "schedule": {
        "type": "exponential_decay",
        "config": {
          "decay_rate": 0.9,
          "decay_steps": 40000,
          "learning_rate": 0.001
        }
      }
    }
  },
  "loss": {
    "hinge": 0.5,
    "cutoff": 255.0,
    "mae_multiplier": 1.0,
    "regularization": 0.01
  },
  "dataset": {
    "batch_size": 16,
    "value_range": [0, 255],
    "scale_range": [0.5, 1.0],
    "clip_value": True,
    "random_blur": True,
    "subsample_size": -1,
    "round_values": True,
    "random_invert": False,
    "random_rotate": 0.314,
    "random_up_down": True,
    "color_mode": "rgb",
    "random_left_right": True,
    "input_shape": [256, 256, 3],
    "multiplicative_noise": [0.05, 0.1, 0.2],
    "additional_noise": [1, 5, 10, 20],
    "inputs": [
      {
        "dataset_shape": [256, 768],
        "directory": "/media/data1_4tb/datasets/KITTI/data/depth/raw_image_values/"
      },
      {
        "dataset_shape": [512, 512],
        "directory": "/media/data1_4tb/datasets/COCO/train2017"
      },
      {
        "dataset_shape": [512, 512],
        "directory": "/media/data1_4tb/datasets/Megadepth/"
      },
      {
        "dataset_shape": [360, 640],
        "directory": "/media/data1_4tb/datasets/bdd_data/train/"
      },
      {
        "dataset_shape": [512, 512],
        "directory": "/media/data1_4tb/datasets/WIDER/WIDER_train/"
      },
      {
        "dataset_shape": [512, 512],
        "directory": "/media/data1_4tb/datasets/WFLW/images/"
      }
    ]
  }
}


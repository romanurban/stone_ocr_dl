# Training Module

This folder contains scripts and entry points for training OCR models on stone inscription datasets.

## Usage

- Run the main training script to start model training with your configuration.
- Training parameters, model type, and data augmentation options are controlled via YAML files in the `configs/` directory.
- Example command:
  ```sh
  python train.py --config ../../../../configs/resnet34.yaml
  ```

## Features
- Supports multiple model architectures (UNet, UNet++, Attention UNet, ResNet, etc.)
- Integrates with PyTorch Lightning for efficient training and logging
- Handles class balancing, encoder freezing, and other advanced training options
- Saves checkpoints and logs to `lightning_logs/`

## Notes
- Adjust configuration files in `configs/` to customize training.
- See the main project README for setup and environment details.

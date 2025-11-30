# CS6886W - Assignment 3 Reference Solution

This repository trains MobileNet-v2 on CIFAR-10 and implements a manual
compression pipeline based on uniform quantization of both weights and
activations. It is designed to match the assignment specification and to be
easy to compare against student submissions.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Tested with:
- Python >= 3.9
- PyTorch >= 2.0
- torchvision >= 0.15
- wandb >= 0.16 (optional, for logging)

## Training the FP32 Baseline

```bash
python train.py         --data-dir ./data         --epochs 200         --batch-size 128         --lr 0.1         --weight-decay 5e-4         --scheduler cosine         --width-mult 1.0         --dropout 0.2         --save-path ./checkpoints/mobilenetv2_cifar10_fp32.pth         --log-wandb         --wandb-project cs6886-assignment3
```

This will:
- Download CIFAR-10.
- Train MobileNet-v2 on 32x32 inputs (first conv stride adapted to 1).
- Save the best checkpoint by test accuracy.
- Optionally log metrics to Weights & Biases.

## Evaluating and Compressing the Model

The main script for compression experiments is `test.py`.

### 1) FP32 Evaluation (no compression)

```bash
python test.py         --checkpoint ./checkpoints/mobilenetv2_cifar10_fp32.pth         --weight_quant_bits 0         --activation_quant_bits 0         --data-dir ./data
```

### 2) Quantized Weights & Activations

```bash
python test.py         --checkpoint ./checkpoints/mobilenetv2_cifar10_fp32.pth         --weight_quant_bits 8         --activation_quant_bits 8         --data-dir ./data         --log-wandb         --wandb-project cs6886-assignment3
```

This will:
- Load the trained MobileNet-v2.
- Build a quantized copy:
  - Conv/Linear weights uniformly quantized to N bits (manual implementation).
  - Activation quantization modules injected after Conv/Linear layers.
- Measure:
  - Test accuracy.
  - Baseline FP32 model size (MB).
  - Compressed model size (MB).
  - Compression ratios for model, weights, and activations.
- Log everything as a W&B run. A W&B Parallel Coordinates plot can then be
  generated over multiple runs with different bit-widths and hyperparameters.

### Example Sweep (manual)

Run several experiments:

```bash
# 4-bit weights, 8-bit activations
python test.py --checkpoint checkpoints/mobilenetv2_cifar10_fp32.pth         --weight_quant_bits 4 --activation_quant_bits 8 --data-dir ./data --log-wandb

# 8-bit weights, 4-bit activations
python test.py --checkpoint checkpoints/mobilenetv2_cifar10_fp32.pth         --weight_quant_bits 8 --activation_quant_bits 4 --data-dir ./data --log-wandb

# 6-bit weights, 6-bit activations
python test.py --checkpoint checkpoints/mobilenetv2_cifar10_fp32.pth         --weight_quant_bits 6 --activation_quant_bits 6 --data-dir ./data --log-wandb
```

On the W&B UI, select a project and use the Parallel Coordinates visualization
to plot:
- `test_acc_quantized`
- `model_compression_ratio`
- `weight_quant_bits`
- `activation_quant_bits`
- any other hyperparameters you logged during training/testing.

## Notes

- No `torch.quantization` or other compression APIs are used; all quantization
  and size accounting are implemented manually.
- MobileNet-v2 is imported from torchvision and adapted for CIFAR-10 by:
  - Changing the first conv stride to 1 for 32x32 inputs.
  - Replacing the classifier head to output 10 classes.

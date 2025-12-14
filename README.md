# MobileNetV2 CIFAR-10 (FP32) + Manual Quantization

Train an **FP32** MobileNetV2 baseline on **CIFAR‑10**, then evaluate **manual** (custom) quantization of **weights** and/or **activations** and report compression ratios.

> Commands below are **validated against the scripts in this repo** and use **relative paths**.

---

## What’s in the repo

```
.
├── train.py                 # Train FP32 baseline (saves best checkpoint)
├── wandbtrain.py            # Same as train.py (alias for convenience)
├── test.py                  # Evaluate FP32 + optional quantization + compression stats
├── data.py                  # CIFAR-10 loaders/transforms
├── models/
│   └── mobilenetv2_cifar.py # MobileNetV2 adapted for CIFAR-10
├── quantization.py          # Activation quant utils + observer
├── compression.py           # Weight quant + 2/4-bit packing + size accounting
├── sweeps/
│   └── quant_sweep.yaml     # W&B sweep for test.py (edit paths to relative)
├── requirements.txt
└── *.ipynb                  # Example notebook with run commands
```

---

## Setup

From the folder that contains `train.py`:

```bash
python -m venv .venv
source .venv/bin/activate    # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CIFAR‑10 downloads automatically to `./data/` on first run.

---

## The commands you run

### 1) Train (FP32)

**Quick smoke run (1 epoch)** (matches the notebook):

```bash
python train.py \
  --data-dir ./data \
  --epochs 1 \
  --batch-size 128 \
  --lr 0.1 \
  --weight-decay 5e-4 \
  --scheduler cosine \
  --width-mult 1.0 \
  --dropout 0.2 \
  --pretrained \
  --save-path ./checkpoints/mobilenetv2_cifar10_fp32_pretrained.pth
```

**Full baseline run (300 epochs + W&B logging)** (matches the notebook; `wandbtrain.py` == `train.py`):

```bash
python wandbtrain.py \
  --data-dir ./data \
  --batch-size 128 \
  --epochs 300 \
  --lr 0.03 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler cosine \
  --warmup-epochs 5 \
  --width-mult 1.0 \
  --dropout 0.0 \
  --pretrained \
  --autoaugment \
  --random-erasing-prob 0.25 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.2 \
  --cutmix-alpha 1.0 \
  --ema-decay 0.999 \
  --save-path ./checkpoints/mnv2_cifar10_fp32_best.pth \
  --log-wandb \
  --wandb-project cs6886-assignment3 \
  --run-name baseline_best_pretrained
```

Output:
- Best checkpoint is saved to `./checkpoints/...pth` (directory auto-created).

---

### 2) Evaluate FP32 checkpoint

```bash
python test.py \
  --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth \
  --data-dir ./data \
  --batch-size 256
```

---

### 3) Quantize + evaluate (post-training)

**Weights only**:

```bash
# 8-bit weights
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --weight_quant_bits 8

# 4-bit weights (recommended per-channel)
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --weight_quant_bits 4 --weight-per-channel

# 2-bit weights (per-channel strongly recommended)
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --weight_quant_bits 2 --weight-per-channel
```

**Activations only** (fake-quant):

```bash
python test.py \
  --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth \
  --data-dir ./data \
  --activation_quant_bits 8
```

**Weights + activations**:

```bash
python test.py \
  --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth \
  --data-dir ./data \
  --weight_quant_bits 4 \
  --activation_quant_bits 4 \
  --weight-per-channel
```

Notes:
- For **2/4-bit weights**, packing is **ON by default** for real memory savings.
- To disable packing (debugging): add `--no-pack-weights`.

---

## W&B (optional)

### Login

```bash
pip install wandb
wandb login
```

### Quantization sweep (matches the notebook)

1) The repo includes `sweeps/quant_sweep.yaml`, but it may contain **absolute Colab paths**.

Edit these two fields to be **relative**:

- `--checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth`
- `--data-dir ./data`

2) Run the sweep:

```bash
# (optional) ensure folder exists
mkdir -p sweeps

wandb sweep --project cs6886-assignment3 sweeps/quant_sweep.yaml
```

3) Start the agent (grid is 3x3 = 9 runs):

```bash
wandb agent --count 9 <entity>/cs6886-assignment3/<sweep_id>
```

---

## Quick sanity / repo view (optional)

If you want a folder tree (Linux/Colab):

```bash
# Ubuntu/Colab only
sudo apt-get update && sudo apt-get install -y tree
tree -L 3 .
```

---

## Validation note (from the notebook)

If you see commands using flags like `--sweep-method` or `--sweep-runs` for `train.py`/`wandbtrain.py`: those flags are **not supported** by this repo’s `argparse`.

Use **W&B sweep YAML + `wandb sweep` / `wandb agent`** (shown above) instead.

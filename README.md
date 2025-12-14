# MobileNetV2 on CIFAR-10 (FP32 Baseline + Manual Quantization)

This repository trains an **FP32 MobileNet-V2** model on **CIFAR-10** and evaluates a **manual (custom) post-training quantization pipeline** for **weights and activations**.
All quantization, packing, and size accounting logic is implemented **from scratch**, without using PyTorch quantization APIs.

The project reports:
- FP32 baseline accuracy and in-memory size
- True in-memory compression ratios for quantized weights
- Accuracy–compression trade-offs across multiple bit-widths
- W&B-logged grid sweep results

---

## Repository Structure

```
.
├── train.py
├── test.py
├── data.py
├── models/
│   └── mobilenetv2_cifar.py
├── quantization.py
├── compression.py
├── sweeps/
│   └── quant_sweep.yaml
├── requirements.txt
└── *.ipynb
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 downloads automatically to `./data/`.

---

## Usage

### Train FP32 Baseline

```bash
python train.py   --data-dir ./data   --batch-size 128   --epochs 300   --lr 0.03   --scheduler cosine   --pretrained   --save-path ./checkpoints/mnv2_cifar10_fp32_best.pth   --log-wandb
```

---

### Evaluate FP32

```bash
python test.py   --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth   --data-dir ./data
```

---

### Quantization Evaluation

**Weights only**
```bash
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --weight_quant_bits 8
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --weight_quant_bits 6 --weight-per-channel
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --weight_quant_bits 4 --weight-per-channel
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --weight_quant_bits 2 --weight-per-channel
```

**Activations (fake-quant)**
```bash
python test.py --checkpoint ./checkpoints/mnv2_cifar10_fp32_best.pth --data-dir ./data --activation_quant_bits 8
```

---

## W&B Grid Sweep

- Weight bits: {2,4,6,8}
- Activation bits: {2,4,6,8}
- Batch size: {128,256}
- Total runs: 32
- Metric: `test_acc_quantized`

```bash
wandb sweep sweeps/quant_sweep.yaml
wandb agent <entity>/<project>/<sweep_id>
```

---

## Expected Accuracy Ranges

| Configuration | Accuracy |
|--------------|----------|
| FP32 | ~94–95% |
| w=8,a=8 | ~93–94% |
| w=6,a=8 | ~91–93% |
| w=4,a=8 | ~88–91% |
| a≤4 | ~10–60% |

---

## Reproducibility Checklist

- YAML uses relative paths
- Metric name matches logging
- 32-run sweep documented
- 6-bit quantization included
- Batch size sweep documented
- Weight packing enabled
- Activation quantization marked as fake-quant

---

## Validation Note

Unsupported flags like `--sweep-method` are not used. W&B sweeps are executed via YAML only.

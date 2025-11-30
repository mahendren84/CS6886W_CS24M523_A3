import argparse
import copy

import torch
import torch.nn as nn

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(x, *args, **kwargs):
        return x

try:
    import wandb
except ImportError:
    wandb = None

from data import get_cifar10_dataloaders
from models.mobilenetv2_cifar import MobileNetV2CIFAR10
from quantization import default_activation_observer
from compression import (
    quantize_model_weights,
    add_activation_quantization,
    count_float_model_bits,
    count_float_weight_bits,
    count_compressed_model_bits,
    bits_to_mb,
    get_activation_compression_stats,
)
from utils import set_seed, top1_accuracy


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, desc: str):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_acc += top1_accuracy(outputs, targets) * batch_size
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples
    return avg_loss, avg_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate and compress MobileNet-v2 on CIFAR-10 "
            "(manual quantization of weights & activations)"
        )
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, required=True)

    # Quantization config (matching assignment style)
    parser.add_argument(
        "--weight_quant_bits", type=int, default=0,
        help="Bit-width for weights (0 -> no weight quantization)"
    )
    parser.add_argument(
        "--activation_quant_bits", type=int, default=0,
        help="Bit-width for activations (0 -> no activation quantization)"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # W&B logging
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str,
                        default="cs6886-assignment3")
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    # Data
    _, _, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=0.0,
    )

    # Load checkpoint & rebuild baseline model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("config", {})
    width_mult = cfg.get("width_mult", 1.0)
    dropout = cfg.get("dropout", 0.2)

    base_model = MobileNetV2CIFAR10(
        width_mult=width_mult,
        dropout=dropout,
        pretrained=False,
    ).to(device)
    base_model.load_state_dict(ckpt["model_state"])

    # Baseline FP32 evaluation
    baseline_loss, baseline_acc = evaluate(
        base_model, test_loader, device, desc="Baseline FP32"
    )
    baseline_model_bits = count_float_model_bits(base_model)
    baseline_model_size_mb = bits_to_mb(baseline_model_bits)
    baseline_weight_bits = count_float_weight_bits(base_model)

    print(
        f"[Baseline FP32] loss={baseline_loss:.4f}, "
        f"acc={baseline_acc:.2f}%, "
        f"model_size={baseline_model_size_mb:.3f} MB"
    )

    # Optionally just run baseline and exit if no quantization requested
    if args.weight_quant_bits <= 0 and args.activation_quant_bits <= 0:
        # Optionally log to W&B
        use_wandb = args.log_wandb and (wandb is not None)
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config={
                    **cfg,
                    **vars(args),
                    "baseline_model_bits": baseline_model_bits,
                    "baseline_model_size_mb": baseline_model_size_mb,
                },
            )
            wandb.log(
                {
                    "test_loss_baseline": baseline_loss,
                    "test_acc_baseline": baseline_acc,
                }
            )
        return

    # Build quantized model
    quant_model = copy.deepcopy(base_model)

    if args.weight_quant_bits > 0:
        quant_model = quantize_model_weights(
            quant_model, args.weight_quant_bits
        )

    if args.activation_quant_bits > 0:
        default_activation_observer.reset(baseline_bits_per_element=32)
        quant_model = add_activation_quantization(
            quant_model, args.activation_quant_bits
        )
    else:
        # No activation quantization, but still reset observer
        default_activation_observer.reset(baseline_bits_per_element=32)

    # Evaluate quantized model
    q_loss, q_acc = evaluate(
        quant_model, test_loader, device, desc="Quantized model"
    )

    # Compression statistics
    if args.weight_quant_bits > 0:
        compressed_stats = count_compressed_model_bits(
            quant_model, args.weight_quant_bits
        )
        compressed_model_bits = compressed_stats["total_bits"]
        compressed_model_size_mb = bits_to_mb(compressed_model_bits)
        compressed_weight_bits = compressed_stats["weight_bits"]
    else:
        # If weights are not quantized, compressed model is same as baseline
        compressed_model_bits = baseline_model_bits
        compressed_model_size_mb = baseline_model_size_mb
        compressed_weight_bits = baseline_weight_bits

    model_compression_ratio = (
        baseline_model_bits / compressed_model_bits
        if compressed_model_bits > 0
        else 1.0
    )

    if args.weight_quant_bits > 0 and compressed_weight_bits > 0:
        weight_compression_ratio = (
            baseline_weight_bits / compressed_weight_bits
        )
    else:
        weight_compression_ratio = 1.0

    # Activation compression
    if args.activation_quant_bits > 0:
        act_stats = get_activation_compression_stats()
        activation_compression_ratio = act_stats["compression_ratio"]
    else:
        act_stats = {
            "total_elements": 0.0,
            "baseline_bits": 0.0,
            "compressed_bits": 0.0,
            "compression_ratio": 1.0,
        }
        activation_compression_ratio = 1.0

    print(
        f"[Quantized] loss={q_loss:.4f}, acc={q_acc:.2f}% "
        f"(weight_bits={args.weight_quant_bits}, "
        f"activation_bits={args.activation_quant_bits})"
    )
    print(
        f"Baseline model size: {baseline_model_size_mb:.3f} MB "
        f"({baseline_model_bits:.0f} bits)"
    )
    print(
        f"Compressed model size: {compressed_model_size_mb:.3f} MB "
        f"({compressed_model_bits:.0f} bits)"
    )
    print(
        f"Model compression ratio (baseline/compressed): "
        f"{model_compression_ratio:.3f}"
    )
    print(
        f"Weight compression ratio (baseline/compressed weights): "
        f"{weight_compression_ratio:.3f}"
    )
    print(
        f"Activation compression ratio (baseline/compressed): "
        f"{activation_compression_ratio:.3f}"
    )

    # W&B logging for parallel coordinates
    use_wandb = args.log_wandb and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                **cfg,
                **vars(args),
                "baseline_model_bits": baseline_model_bits,
                "baseline_model_size_mb": baseline_model_size_mb,
            },
        )
        wandb.log(
            {
                # Accuracy metrics
                "test_loss_baseline": baseline_loss,
                "test_acc_baseline": baseline_acc,
                "test_loss_quantized": q_loss,
                "test_acc_quantized": q_acc,
                # Compression metrics
                "model_compression_ratio": model_compression_ratio,
                "weight_compression_ratio": weight_compression_ratio,
                "activation_compression_ratio": activation_compression_ratio,
                "compressed_model_bits": compressed_model_bits,
                "compressed_model_size_mb": compressed_model_size_mb,
                # Raw activation stats
                "activation_total_elements": act_stats["total_elements"],
                "activation_baseline_bits": act_stats["baseline_bits"],
                "activation_compressed_bits": act_stats["compressed_bits"],
            }
        )


if __name__ == "__main__":
    main()

import argparse
import copy
import math
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

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
from utils import set_seed, top1_accuracy, save_checkpoint


def _set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def cosine_warmup_lr(
    epoch: int,
    base_lr: float,
    total_epochs: int,
    warmup_epochs: int = 0,
) -> float:
    """Cosine LR schedule with optional linear warmup.

    The schedule is defined over epochs (1-indexed):
      - Warmup: epochs 1..warmup_epochs linearly increase from 0 -> base_lr.
      - Cosine: remaining epochs decay from base_lr -> 0.

    This implementation intentionally avoids relying on scheduler.step() order
    (which is a common source of off-by-one issues).
    """

    if total_epochs <= 1:
        return float(base_lr)

    warmup_epochs = int(max(0, warmup_epochs))
    warmup_epochs = min(warmup_epochs, total_epochs)

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * float(epoch) / float(warmup_epochs)

    # Cosine phase length
    cosine_epochs = total_epochs - warmup_epochs
    if cosine_epochs <= 1:
        return float(base_lr)

    # t goes 0..cosine_epochs-1
    t = epoch - warmup_epochs - 1 if warmup_epochs > 0 else epoch - 1
    t = max(0, min(t, cosine_epochs - 1))
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * t / (cosine_epochs - 1)))


def build_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict]:
    """Build optimizer param groups that exclude BN/bias from weight decay.

    This is a standard CNN training tweak for CIFAR that often improves accuracy.
    """

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Heuristic rules:
        # - 1D parameters are typically BN/Norm scales or biases.
        # - Explicit biases should not be decayed.
        # - Any normalization layer parameters should not be decayed.
        if (
            param.ndim == 1
            or name.endswith(".bias")
            or "bn" in name.lower()
            or "norm" in name.lower()
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _one_hot(
    targets: torch.Tensor, num_classes: int, label_smoothing: float
) -> torch.Tensor:
    """One-hot with optional label smoothing."""
    with torch.no_grad():
        smoothing = float(max(0.0, min(1.0, label_smoothing)))
        off_value = smoothing / float(num_classes)
        on_value = 1.0 - smoothing + off_value

        out = torch.full(
            (targets.size(0), num_classes),
            off_value,
            device=targets.device,
            dtype=torch.float32,
        )
        out.scatter_(1, targets.unsqueeze(1), on_value)
        return out


def soft_cross_entropy(
    logits: torch.Tensor, soft_targets: torch.Tensor
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def rand_bbox(
    width: int, height: int, lam: float
) -> Tuple[int, int, int, int]:
    """Sample CutMix bbox."""
    cut_rat = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)

    cx = random.randint(0, width)
    cy = random.randint(0, height)

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_mixup_or_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    prob: float,
    switch_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp/CutMix and return (images, soft_targets).

    If neither mixup_alpha nor cutmix_alpha are >0, this returns original images
    and smoothed one-hot targets.
    """

    # Default: no mixing
    soft_targets = _one_hot(targets, num_classes, label_smoothing)

    if prob <= 0.0:
        return images, soft_targets

    do_mix = (mixup_alpha and mixup_alpha > 0.0) or (
        cutmix_alpha and cutmix_alpha > 0.0
    )
    if not do_mix:
        return images, soft_targets

    if random.random() > prob:
        return images, soft_targets

    # Choose between MixUp and CutMix
    use_cutmix = False
    if mixup_alpha > 0.0 and cutmix_alpha > 0.0:
        use_cutmix = random.random() < switch_prob
    elif cutmix_alpha > 0.0:
        use_cutmix = True

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    if use_cutmix:
        lam = float(
            torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample().item()
        )
        x1, y1, x2, y2 = rand_bbox(images.size(3), images.size(2), lam)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

        # Adjust lambda based on the actually mixed area.
        box_area = float((x2 - x1) * (y2 - y1))
        lam = 1.0 - box_area / float(images.size(2) * images.size(3))

        y_a = _one_hot(targets, num_classes, label_smoothing)
        y_b = _one_hot(targets[index], num_classes, label_smoothing)
        soft_targets = lam * y_a + (1.0 - lam) * y_b
        return mixed, soft_targets

    # MixUp
    lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample().item())
    mixed = lam * images + (1.0 - lam) * images[index]
    y_a = _one_hot(targets, num_classes, label_smoothing)
    y_b = _one_hot(targets[index], num_classes, label_smoothing)
    soft_targets = lam * y_a + (1.0 - lam) * y_b
    return mixed, soft_targets


class ModelEMA:
    """Exponential Moving Average of model parameters.

    This is a training-time trick that often improves final test accuracy.
    We keep EMA weights in FP32 and evaluate using the EMA copy.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k not in msd:
                continue
            model_v = msd[k]
            if not torch.is_tensor(model_v):
                continue

            # Buffers (e.g., BN running stats) are copied directly.
            if "running_" in k or "num_batches_tracked" in k:
                v.copy_(model_v)
                continue

            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(model_v, alpha=1.0 - self.decay)
            else:
                v.copy_(model_v)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    epoch: int,
    *,
    num_classes: int,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    mixup_switch_prob: float,
    ema: Optional[ModelEMA] = None,
):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, targets in tqdm(
        loader, desc=f"Train epoch {epoch}", leave=False
    ):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # MixUp / CutMix (optional)
        mixed_images, soft_targets = apply_mixup_or_cutmix(
            images,
            targets,
            num_classes=num_classes,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            prob=mixup_prob,
            switch_prob=mixup_switch_prob,
        )

        optimizer.zero_grad(set_to_none=True)
        outputs = model(mixed_images)
        loss = soft_cross_entropy(outputs, soft_targets)
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        # Note: with MixUp/CutMix, this "train_acc" is only indicative since
        # targets are mixed. We still compute it against the original labels
        # to track training progress.
        running_acc += top1_accuracy(outputs, targets) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    desc: str = "Test",
):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, targets in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += top1_accuracy(outputs, targets) * batch_size
        total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_acc / total_samples
    return epoch_loss, epoch_acc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNet-v2 on CIFAR-10 (FP32 baseline)"
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "none"])
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Linear warmup epochs (0 disables warmup).",
    )
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true")

    # Data recipe
    parser.add_argument(
        "--norm",
        type=str,
        default=None,
        choices=["cifar", "imagenet"],
        help=(
            "Normalization to use. If omitted, defaults to 'imagenet' when "
            "--pretrained is set, else 'cifar'."
        ),
    )
    parser.add_argument(
        "--autoaugment",
        action="store_true",
        help="Enable AutoAugment(CIFAR10) for training.",
    )
    parser.add_argument(
        "--random-erasing-prob",
        type=float,
        default=0.0,
        help="RandomErasing probability (0 disables).",
    )

    # Loss / regularization
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0 disables).",
    )

    # MixUp / CutMix
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.0,
        help="MixUp Beta(alpha, alpha). 0 disables.",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=0.0,
        help="CutMix Beta(alpha, alpha). 0 disables.",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of applying MixUp/CutMix on a batch.",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="If both MixUp and CutMix are enabled, probability of using CutMix.",
    )

    # EMA
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="EMA decay (0 disables EMA). Typical: 0.999.",
    )
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-path", type=str,
                        default="./checkpoints/mobilenetv2_cifar10_fp32.pth")
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

    # Decide normalization (important when --pretrained is used)
    chosen_norm = args.norm or ("imagenet" if args.pretrained else "cifar")

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        norm=chosen_norm,
        pretrained=args.pretrained,
        use_autoaugment=args.autoaugment,
        random_erasing_prob=args.random_erasing_prob,
    )

    model = MobileNetV2CIFAR10(
        width_mult=args.width_mult,
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)

    # Optimizer with no weight decay on BN/bias
    param_groups = build_param_groups(model, args.weight_decay)
    optimizer = optim.SGD(
        param_groups,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    # EMA (optional)
    ema: Optional[ModelEMA] = None
    if args.ema_decay and args.ema_decay > 0.0:
        ema = ModelEMA(model, decay=args.ema_decay)

    # W&B
    use_wandb = args.log_wandb and (wandb is not None)
    if use_wandb:
        wb_cfg = vars(args).copy()
        wb_cfg["chosen_norm"] = chosen_norm
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=wb_cfg,
        )

    num_classes = 10
    eval_criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # LR schedule (epoch-level)
        if args.scheduler == "cosine":
            lr = cosine_warmup_lr(
                epoch=epoch,
                base_lr=args.lr,
                total_epochs=args.epochs,
                warmup_epochs=args.warmup_epochs,
            )
            _set_optimizer_lr(optimizer, lr)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            num_classes=num_classes,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
            mixup_switch_prob=args.mixup_switch_prob,
            ema=ema,
        )

        # Evaluate on the EMA copy if enabled
        eval_model = ema.ema if ema is not None else model

        if val_loader is not None:
            val_loss, val_acc = evaluate(
                eval_model, val_loader, eval_criterion, device, desc="Val"
            )
        else:
            val_loss, val_acc = 0.0, 0.0

        test_loss, test_acc = evaluate(
            eval_model, test_loader, eval_criterion, device, desc="Test"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            # Save EMA weights if enabled (these are the weights we evaluated).
            save_model = eval_model
            train_cfg = vars(args).copy()
            train_cfg["norm"] = chosen_norm
            save_checkpoint(
                model=save_model,
                path=args.save_path,
                epoch=epoch,
                best_acc=best_test_acc,
                train_config=train_cfg,
            )

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}, "
            f"best_test_acc={best_test_acc:.2f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "best_test_acc": best_test_acc,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

    print(f"Training done. Best test accuracy: {best_test_acc:.2f}%")


if __name__ == "__main__":
    main()

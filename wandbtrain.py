import argparse
from types import SimpleNamespace

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


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0

    for images, targets in tqdm(loader, desc=f"Train epoch {epoch}", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
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


def run_training(config, device, use_wandb: bool = False):
    """
    Core training loop used for both single-run training and W&B sweeps.

    `config` is expected to have attributes:
      data_dir, batch_size, num_workers, val_split,
      width_mult, dropout, pretrained,
      lr, momentum, weight_decay, scheduler, epochs,
      save_path, seed
    """
    set_seed(int(config.seed))

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=config.data_dir,
        batch_size=int(config.batch_size),
        num_workers=int(config.num_workers),
        val_split=float(config.val_split),
    )

    model = MobileNetV2CIFAR10(
        width_mult=float(config.width_mult),
        dropout=float(config.dropout),
        pretrained=bool(config.pretrained),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(config.lr),
        momentum=float(config.momentum),
        weight_decay=float(config.weight_decay),
        nesterov=True,
    )

    if config.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(config.epochs)
        )
    else:
        scheduler = None

    best_test_acc = 0.0
    best_epoch = 0

    for epoch in range(1, int(config.epochs) + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        if val_loader is not None:
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, desc="Val"
            )
        else:
            val_loss, val_acc = 0.0, 0.0

        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, desc="Test"
        )

        if scheduler is not None:
            scheduler.step()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            save_checkpoint(
                model=model,
                path=config.save_path,
                epoch=epoch,
                best_acc=best_test_acc,
                train_config=vars(config)
                if isinstance(config, argparse.Namespace)
                else dict(config.__dict__),
            )

        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}, "
            f"test_loss={test_loss:.4f}, test_acc={test_acc:.2f}, "
            f"best_test_acc={best_test_acc:.2f} (epoch {best_epoch})"
        )

        if use_wandb and wandb is not None:
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
    return best_test_acc


def build_sweep_config(method: str):
    """
    Build a W&B sweep configuration for either bayesian search
    or bayesian search with Hyperband early termination.

    method: "bayes" or "hyperband"
    """
    sweep_config = {
        "name": f"mobilenetv2_cifar10_{method}_sweep",
        "metric": {"name": "best_test_acc", "goal": "maximize"},
        # We keep the underlying search method as 'bayes' for both,
        # and enable Hyperband as an early-termination strategy when requested.
        "method": "bayes",
        "parameters": {
            "lr": {
                "distribution": "log_uniform_values",
                "min": 1e-4,
                "max": 1e-1,
            },
            "batch_size": {"values": [64, 128, 256]},
            "weight_decay": {
                "distribution": "log_uniform_values",
                "min": 1e-5,
                "max": 5e-4,
            },
            "momentum": {"values": [0.85, 0.9, 0.95]},
            "width_mult": {"values": [0.75, 1.0, 1.25]},
            "dropout": {"values": [0.1, 0.2, 0.3]},
            "epochs": {"values": [100, 150, 200]},
            "scheduler": {"values": ["cosine", "none"]},
        },
    }

    if method == "hyperband":
        sweep_config["early_terminate"] = {
            "type": "hyperband",
            "min_iter": 20,
            "max_iter": 200,
            "eta": 3,
        }

    return sweep_config


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train MobileNet-v2 on CIFAR-10 (FP32 baseline) with optional W&B sweeps"
        )
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "none"]
    )
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--save-path",
        type=str,
        default="./checkpoints/mobilenetv2_cifar10_fp32.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    # Sweep-related args
    parser.add_argument(
        "--sweep-method",
        type=str,
        default="none",
        choices=["none", "bayes", "hyperband"],
        help="If not 'none', run a W&B sweep with this method.",
    )
    parser.add_argument(
        "--sweep-runs",
        type=int,
        default=18,
        help="Number of runs to execute in the sweep (for bayes/hyperband).",
    )

    # W&B logging
    parser.add_argument("--log-wandb", action="store_true")
    parser.add_argument(
        "--wandb-project", type=str, default="cs6886-assignment3"
    )
    parser.add_argument("--run-name", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # Device selection
    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    # If no sweep: behave like a normal training script
    if args.sweep_method == "none":
        use_wandb = args.log_wandb and (wandb is not None)
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )
        run_training(args, device, use_wandb=use_wandb)
        return

    # Otherwise, set up a W&B sweep (bayes or hyperband)
    if wandb is None:
        raise RuntimeError(
            "wandb is not installed, but --sweep-method was specified. "
            "Install wandb (`pip install wandb`) to use sweeps."
        )

    sweep_config = build_sweep_config(args.sweep_method)
    sweep_id = wandb.sweep(
        sweep_config, project=args.wandb_project
    )

    print(
        f"Created W&B sweep with id={sweep_id}, "
        f"method={args.sweep_method}, runs={args.sweep_runs}"
    )

    def sweep_train():
        # One training run inside the sweep
        with wandb.init(project=args.wandb_project) as run:
            # Combine CLI defaults (args) with sweep-sampled values (wandb.config)
            cfg_dict = vars(args).copy()
            for key, value in wandb.config.items():
                cfg_dict[key] = value

            # Use a unique checkpoint file per run so they don't overwrite
            run_id = run.name or run.id
            cfg_dict["save_path"] = (
                f"./checkpoints/mobilenetv2_cifar10_fp32_{run_id}.pth"
            )

            config = SimpleNamespace(**cfg_dict)
            best_acc = run_training(config, device, use_wandb=True)
            run.summary["best_test_acc"] = best_acc

    wandb.agent(sweep_id, function=sweep_train, count=args.sweep_runs)


if __name__ == "__main__":
    main()

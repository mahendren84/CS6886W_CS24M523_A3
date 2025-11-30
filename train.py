import argparse

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

    for images, targets in tqdm(
        loader, desc=f"Train epoch {epoch}", leave=False
    ):
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
    parser.add_argument("--width-mult", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true")
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

    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    model = MobileNetV2CIFAR10(
        width_mult=args.width_mult,
        dropout=args.dropout,
        pretrained=args.pretrained,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs
        )
    else:
        scheduler = None

    # W&B
    use_wandb = args.log_wandb and (wandb is not None)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
        )

    best_test_acc = 0.0

    for epoch in range(1, args.epochs + 1):
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
            save_checkpoint(
                model=model,
                path=args.save_path,
                epoch=epoch,
                best_acc=best_test_acc,
                train_config=vars(args),
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

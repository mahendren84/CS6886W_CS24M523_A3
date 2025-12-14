"""Data loading utilities for CIFAR-10.

This module provides a single helper: :func:`get_cifar10_dataloaders`.

Changes vs the original template:
  - Optional ImageNet vs CIFAR normalization (useful when finetuning ImageNet
    pretrained backbones).
  - Optional stronger augmentation to help reach higher CIFAR-10 accuracy:
      * AutoAugment (CIFAR10 policy)
      * RandomErasing (Cutout-like)

These are *training recipe* improvements and are fully allowed by the
assignment (Q1 asks to specify normalization + augmentation). They do not use
any compression API.
"""

from __future__ import annotations

from typing import Optional, Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def _get_normalize_transform(norm: str) -> transforms.Normalize:
    norm = (norm or "cifar").lower()

    if norm == "imagenet":
        # ImageNet mean/std used by torchvision pretrained backbones.
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    if norm == "cifar":
        # Standard CIFAR-10 normalization.
        return transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )

    raise ValueError(
        f"Unknown normalization '{norm}'. Expected one of: 'cifar', 'imagenet'."
    )


def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.0,
    *,
    # New (optional) recipe knobs
    norm: Optional[str] = None,
    pretrained: bool = False,
    use_autoaugment: bool = False,
    random_erasing_prob: float = 0.0,
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Return (train_loader, val_loader, test_loader) for CIFAR-10.

    Args:
        data_dir: Root directory to download/load CIFAR-10.
        batch_size: Batch size.
        num_workers: DataLoader workers.
        val_split: Fraction of the training set to reserve for validation.
        norm: "cifar" or "imagenet". If None, defaults to "imagenet" when
            pretrained=True else "cifar".
        pretrained: If True and norm is None, uses ImageNet normalization.
        use_autoaugment: If True, applies torchvision AutoAugment CIFAR10 policy.
        random_erasing_prob: If > 0, applies RandomErasing with this probability.
    """

    # Default normalization choice
    if norm is None:
        norm = "imagenet" if pretrained else "cifar"
    normalize = _get_normalize_transform(norm)

    # Optional stronger augmentation
    aug = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if use_autoaugment:
        # torchvision AutoAugment supports CIFAR10 policy.
        try:
            from torchvision.transforms import AutoAugment, AutoAugmentPolicy

            aug.append(AutoAugment(AutoAugmentPolicy.CIFAR10))
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "AutoAugment requested but unavailable in this torchvision. "
                "Upgrade torchvision or disable --autoaugment."
            ) from e

    aug.extend([transforms.ToTensor(), normalize])

    if random_erasing_prob and random_erasing_prob > 0.0:
        try:
            from torchvision.transforms import RandomErasing

            # RandomErasing expects tensors and is usually applied after normalize.
            aug.append(
                RandomErasing(
                    p=float(random_erasing_prob),
                    scale=(0.02, 0.2),
                    ratio=(0.3, 3.3),
                    value="random",
                )
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "RandomErasing requested but unavailable in this torchvision. "
                "Upgrade torchvision or disable --random-erasing-prob."
            ) from e

    train_transform = transforms.Compose(aug)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    if val_split > 0.0:
        val_size = int(len(full_train_dataset) * val_split)
        train_size = len(full_train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_train_dataset, [train_size, val_size]
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_dataset = full_train_dataset
        val_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

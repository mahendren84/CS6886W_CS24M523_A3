import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_cifar10_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.0,
):
    """
    Returns train, val, test dataloaders for CIFAR-10.
    If val_split == 0, val_loader is None and train uses all training data.
    """
    # Standard CIFAR-10 normalization
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

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

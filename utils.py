import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def top1_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor
) -> float:
    with torch.no_grad():
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
    return 100.0 * correct / total


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int,
    best_acc: float,
    train_config: dict,
) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "epoch": epoch,
        "best_acc": best_acc,
        "config": train_config,
    }
    torch.save(state, path_obj)


def load_checkpoint(
    model_ctor,
    checkpoint_path: str,
    device: torch.device,
    override_args: dict = None,
) -> Tuple[torch.nn.Module, dict]:
    """
    model_ctor: callable that creates a model from kwargs, e.g.
        lambda cfg: MobileNetV2CIFAR10(**cfg)
    override_args: dict to override config keys when rebuilding model
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    if override_args:
        cfg.update(override_args)

    # Only keep keys relevant to model constructor
    allowed_keys = ["width_mult", "dropout", "pretrained"]
    model_kwargs = {k: cfg[k] for k in allowed_keys if k in cfg}
    model = model_ctor(model_kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt

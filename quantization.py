from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class QuantConfig:
    num_bits: int = 8
    symmetric: bool = True


def _int_dtype_for_bits(num_bits: int) -> torch.dtype:
    """Choose a compact integer dtype for a given bit-width."""
    if num_bits <= 8:
        return torch.int8
    if num_bits <= 16:
        return torch.int16
    return torch.int32


def quantize_tensor(
    x: torch.Tensor,
    num_bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uniform symmetric quantization:
    - x: float tensor
    - num_bits: number of bits (e.g., 2, 4, 8)
    Returns:
        q_x: integer tensor with quantized values (int8/int16/int32 depending
            on num_bits)
        scale: scalar float32 tensor
    """
    if num_bits >= 32:
        # No quantization
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        q_x = x.to(torch.int32)
        return q_x, scale

    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    max_val = x.abs().max()
    if max_val == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        q_x = torch.zeros_like(x, dtype=torch.int32)
        return q_x, scale

    scale = max_val / float(qmax)
    q_x = torch.clamp(torch.round(x / scale), qmin, qmax).to(
        _int_dtype_for_bits(num_bits)
    )
    return q_x, scale


def quantize_tensor_per_channel(
    x: torch.Tensor,
    num_bits: int,
    channel_dim: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel symmetric quantization.

    For Conv2d weights (O, I, kH, kW) use channel_dim=0 (per output channel).
    For Linear weights (O, I) use channel_dim=0 (per output feature).
    """

    if num_bits >= 32:
        scale = torch.ones(
            x.size(channel_dim), device=x.device, dtype=torch.float32
        )
        return x.to(torch.int32), scale

    qmin = -(1 << (num_bits - 1))
    qmax = (1 << (num_bits - 1)) - 1

    # Move channel dimension to front, flatten the rest
    x_perm = x.transpose(0, channel_dim).contiguous()
    c = x_perm.shape[0]
    x_flat = x_perm.view(c, -1)

    max_val = x_flat.abs().max(dim=1).values
    # Avoid div by 0
    max_val = torch.where(max_val == 0, torch.ones_like(max_val), max_val)
    scale = (max_val / float(qmax)).to(torch.float32)

    q = torch.round(x_flat / scale[:, None])
    q = torch.clamp(q, qmin, qmax).to(_int_dtype_for_bits(num_bits))
    q = q.view_as(x_perm).transpose(0, channel_dim).contiguous()
    return q, scale


def dequantize_tensor(q_x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize back to float32 using the stored scale.
    """
    return q_x.to(torch.float32) * scale


def dequantize_tensor_per_channel(
    q_x: torch.Tensor,
    scale: torch.Tensor,
    channel_dim: int = 0,
) -> torch.Tensor:
    """Dequantize per-channel quantized tensors."""
    # Broadcast scale along non-channel dims
    shape = [1] * q_x.ndim
    shape[channel_dim] = -1
    return q_x.to(torch.float32) * scale.view(*shape)


class ActivationObserver:
    """
    Tracks approximate activation storage:
    - Counts total elements seen.
    - Uses a fixed baseline bits-per-element (default: 32).
    - Uses the quantized bits-per-element passed by QuantActivation.
    """

    def __init__(self):
        self.total_elements: int = 0
        self.baseline_bits_per_element: int = 32
        self.quant_bits_per_element: Optional[int] = None

    def reset(self, baseline_bits_per_element: int = 32) -> None:
        self.total_elements = 0
        self.baseline_bits_per_element = baseline_bits_per_element
        self.quant_bits_per_element = None

    def record(self, num_elements: int, quant_bits_per_element: int) -> None:
        self.total_elements += int(num_elements)
        self.quant_bits_per_element = quant_bits_per_element

    def summary(self) -> Dict[str, float]:
        if self.total_elements == 0:
            return {
                "total_elements": 0,
                "baseline_bits": 0.0,
                "compressed_bits": 0.0,
                "compression_ratio": 1.0,
            }

        baseline_bits = (
            self.total_elements * self.baseline_bits_per_element
        )
        if self.quant_bits_per_element is None:
            compressed_bits = baseline_bits
        else:
            compressed_bits = (
                self.total_elements * self.quant_bits_per_element
            )

        compression_ratio = (
            baseline_bits / compressed_bits if compressed_bits > 0 else 1.0
        )

        return {
            "total_elements": float(self.total_elements),
            "baseline_bits": float(baseline_bits),
            "compressed_bits": float(compressed_bits),
            "compression_ratio": float(compression_ratio),
        }


# Global observer used by all QuantActivation modules
default_activation_observer = ActivationObserver()


class QuantActivation(nn.Module):
    """
    Fake-quantization module for activations.
    Used only during evaluation of the compressed model.
    """

    def __init__(
        self,
        num_bits: int = 8,
        observer: ActivationObserver = default_activation_observer,
        enabled: bool = True,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.observer = observer
        self.enabled = enabled
        self.name = name or "quant_act"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.num_bits >= 32:
            # Record baseline activations if desired; here we just skip
            return x

        # Quantize + dequantize (fake quantization)
        q_x, scale = quantize_tensor(x, self.num_bits)
        x_hat = dequantize_tensor(q_x, scale)

        # Record activation statistics for compression metrics
        if self.observer is not None:
            self.observer.record(x_hat.numel(), self.num_bits)

        return x_hat

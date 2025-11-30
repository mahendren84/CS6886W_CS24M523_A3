from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization import (
    quantize_tensor,
    dequantize_tensor,
    QuantActivation,
    default_activation_observer,
)

BYTES_IN_MB = 1024.0 * 1024.0


def bits_to_mb(bits: float) -> float:
    return bits / 8.0 / BYTES_IN_MB


class QuantizedConv2d(nn.Module):
    """
    Wrapper around nn.Conv2d with offline uniform quantized weights.
    We store int32 qweight + float32 scale, and keep bias in FP32.
    """

    def __init__(self, conv: nn.Conv2d, num_bits: int):
        super().__init__()
        self.num_bits = num_bits

        qweight, scale = quantize_tensor(conv.weight.data, num_bits)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data.clone())
        else:
            self.bias = None

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_tensor(self.qweight, self.scale)
        return F.conv2d(
            x,
            weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )


class QuantizedLinear(nn.Module):
    """
    Wrapper around nn.Linear with offline uniform quantized weights.
    """

    def __init__(self, linear: nn.Linear, num_bits: int):
        super().__init__()
        self.num_bits = num_bits

        qweight, scale = quantize_tensor(linear.weight.data, num_bits)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scale", scale)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = dequantize_tensor(self.qweight, self.scale)
        return F.linear(x, weight, self.bias)


def quantize_model_weights(model: nn.Module, num_bits: int) -> nn.Module:
    """
    Recursively replace Conv2d and Linear modules with quantized wrappers.
    """

    def _recursive_quantize(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, name, QuantizedConv2d(child, num_bits))
            elif isinstance(child, nn.Linear):
                setattr(module, name, QuantizedLinear(child, num_bits))
            else:
                _recursive_quantize(child)

    _recursive_quantize(model)
    return model


def add_activation_quantization(
    model: nn.Module, num_bits: int
) -> nn.Module:
    """
    Inject QuantActivation modules after Conv2d / Linear (or their quantized
    counterparts) by wrapping them in an nn.Sequential.
    """

    target_types = (QuantizedConv2d, QuantizedLinear, nn.Conv2d, nn.Linear)

    def _wrap_children(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, target_types):
                wrapped = nn.Sequential(
                    child,
                    QuantActivation(num_bits=num_bits),
                )
                setattr(module, name, wrapped)
            else:
                _wrap_children(child)

    _wrap_children(model)
    return model


def count_float_model_bits(
    model: nn.Module, dtype_bits: int = 32
) -> int:
    """
    Count total parameter bits for an FP32 model (no compression).
    """
    total_bits = 0
    for p in model.parameters():
        total_bits += p.numel() * dtype_bits
    return total_bits


def count_float_weight_bits(
    model: nn.Module, dtype_bits: int = 32
) -> int:
    """
    Count bits used by Conv2d and Linear weights only (baseline).
    """
    total_bits = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            total_bits += module.weight.numel() * dtype_bits
    return total_bits


def count_compressed_model_bits(
    model: nn.Module, weight_bits: int
) -> Dict[str, float]:
    """
    For a quantized model using QuantizedConv2d/QuantizedLinear:
    - Computes total bits = quantized weights + scales + remaining FP32 params.
    - Also returns weight_bits_only and scale_bits_only for analysis.
    """

    quant_weight_bits = 0
    scale_bits = 0

    for module in model.modules():
        if isinstance(module, QuantizedConv2d):
            quant_weight_bits += module.qweight.numel() * weight_bits
            scale_bits += module.scale.numel() * 32
        elif isinstance(module, QuantizedLinear):
            quant_weight_bits += module.qweight.numel() * weight_bits
            scale_bits += module.scale.numel() * 32

    float_param_bits = 0
    for p in model.parameters():
        float_param_bits += p.numel() * 32

    total_bits = quant_weight_bits + scale_bits + float_param_bits

    return {
        "total_bits": float(total_bits),
        "weight_bits": float(quant_weight_bits),
        "scale_bits": float(scale_bits),
        "float_param_bits": float(float_param_bits),
    }


def get_activation_compression_stats() -> Dict[str, float]:
    """
    Return activation baseline/compressed bits and compression ratio.
    """
    return default_activation_observer.summary()

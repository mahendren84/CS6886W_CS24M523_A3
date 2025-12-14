from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantization import (
    quantize_tensor,
    dequantize_tensor,
    quantize_tensor_per_channel,
    dequantize_tensor_per_channel,
    QuantActivation,
    default_activation_observer,
)

BYTES_IN_MB = 1024.0 * 1024.0


def bits_to_mb(bits: float) -> float:
    return bits / 8.0 / BYTES_IN_MB


def pack_nbit_tensor(qweight: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Pack a signed integer tensor into a uint8 buffer.

    Supported bit-widths:
      - 4-bit: packs 2 values per byte
      - 2-bit: packs 4 values per byte

    We use two's complement representation for negative values.
    """

    if num_bits not in (2, 4):
        raise ValueError(f"pack_nbit_tensor only supports 2 or 4 bits, got {num_bits}")

    device = qweight.device
    flat = qweight.flatten()
    mask = (1 << num_bits) - 1

    # Convert signed values to low-bit two's complement (stored as unsigned 0..mask)
    q_u = (flat.to(torch.int16) & mask).to(torch.uint8)

    if num_bits == 4:
        # pad to even length
        if q_u.numel() % 2 != 0:
            q_u = torch.cat([q_u, torch.zeros(1, device=device, dtype=torch.uint8)])
        q_u = q_u.view(-1, 2)
        packed = q_u[:, 0] | (q_u[:, 1] << 4)
        return packed.contiguous()

    # num_bits == 2
    pad = (-q_u.numel()) % 4
    if pad:
        q_u = torch.cat(
            [q_u, torch.zeros(pad, device=device, dtype=torch.uint8)], dim=0
        )
    q_u = q_u.view(-1, 4)
    packed = (
        q_u[:, 0]
        | (q_u[:, 1] << 2)
        | (q_u[:, 2] << 4)
        | (q_u[:, 3] << 6)
    )
    return packed.contiguous()


def unpack_nbit_tensor(
    packed: torch.Tensor,
    num_bits: int,
    *,
    numel: int,
    shape: tuple,
) -> torch.Tensor:
    """Unpack a uint8 packed buffer back into a signed int8 tensor."""

    if num_bits not in (2, 4):
        raise ValueError(
            f"unpack_nbit_tensor only supports 2 or 4 bits, got {num_bits}"
        )

    packed = packed.to(torch.uint8).flatten()

    if num_bits == 4:
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        q_u = torch.stack([low, high], dim=1).flatten()[:numel]
        # 4-bit two's complement -> signed
        q_s = q_u.to(torch.int16)
        q_s = torch.where(q_s >= 8, q_s - 16, q_s)
        return q_s.to(torch.int8).view(*shape)

    # num_bits == 2
    q0 = packed & 0x03
    q1 = (packed >> 2) & 0x03
    q2 = (packed >> 4) & 0x03
    q3 = (packed >> 6) & 0x03
    q_u = torch.stack([q0, q1, q2, q3], dim=1).flatten()[:numel]
    q_s = q_u.to(torch.int16)
    # 2-bit two's complement: 0->0, 1->1, 2->-2, 3->-1
    q_s = torch.where(q_s >= 2, q_s - 4, q_s)
    return q_s.to(torch.int8).view(*shape)


class QuantizedConv2d(nn.Module):
    """
    Wrapper around nn.Conv2d with offline uniform quantized weights.

    Storage:
      - 8-bit weights are stored as int8 buffers.
      - 4-bit/2-bit weights can be stored packed into uint8 buffers
        (real in-memory savings).
      - Scales are stored as float32 (scalar or per-channel).
      - Bias (if any) is kept in FP32.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        num_bits: int,
        *,
        per_channel: bool = False,
        pack: bool = True,
    ):
        super().__init__()
        self.num_bits = num_bits

        self.per_channel = bool(per_channel)
        self.pack = bool(pack)

        if self.per_channel:
            qweight, scale = quantize_tensor_per_channel(
                conv.weight.data, num_bits, channel_dim=0
            )
        else:
            qweight, scale = quantize_tensor(conv.weight.data, num_bits)

        # Store weights in a compact form when possible.
        # - For 8-bit: store as int8 directly.
        # - For 4-bit/2-bit: pack into uint8 buffers (real "in-memory" savings).
        if self.pack and num_bits in (2, 4):
            packed = pack_nbit_tensor(qweight, num_bits)
            self.register_buffer("qweight_packed", packed)
            self.qweight_shape = tuple(qweight.shape)
            self.qweight_numel = int(qweight.numel())
        else:
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
        # Unpack if needed
        if hasattr(self, "qweight_packed"):
            qweight = unpack_nbit_tensor(
                self.qweight_packed,
                self.num_bits,
                numel=self.qweight_numel,
                shape=self.qweight_shape,
            )
        else:
            qweight = self.qweight

        if self.per_channel:
            weight = dequantize_tensor_per_channel(qweight, self.scale, channel_dim=0)
        else:
            weight = dequantize_tensor(qweight, self.scale)
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

    def __init__(
        self,
        linear: nn.Linear,
        num_bits: int,
        *,
        per_channel: bool = False,
        pack: bool = True,
    ):
        super().__init__()
        self.num_bits = num_bits

        self.per_channel = bool(per_channel)
        self.pack = bool(pack)

        if self.per_channel:
            qweight, scale = quantize_tensor_per_channel(
                linear.weight.data, num_bits, channel_dim=0
            )
        else:
            qweight, scale = quantize_tensor(linear.weight.data, num_bits)

        if self.pack and num_bits in (2, 4):
            packed = pack_nbit_tensor(qweight, num_bits)
            self.register_buffer("qweight_packed", packed)
            self.qweight_shape = tuple(qweight.shape)
            self.qweight_numel = int(qweight.numel())
        else:
            self.register_buffer("qweight", qweight)

        self.register_buffer("scale", scale)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "qweight_packed"):
            qweight = unpack_nbit_tensor(
                self.qweight_packed,
                self.num_bits,
                numel=self.qweight_numel,
                shape=self.qweight_shape,
            )
        else:
            qweight = self.qweight

        if self.per_channel:
            weight = dequantize_tensor_per_channel(qweight, self.scale, channel_dim=0)
        else:
            weight = dequantize_tensor(qweight, self.scale)
        return F.linear(x, weight, self.bias)


def quantize_model_weights(
    model: nn.Module,
    num_bits: int,
    *,
    per_channel: bool = False,
    pack: bool = True,
) -> nn.Module:
    """
    Recursively replace Conv2d and Linear modules with quantized wrappers.
    """

    def _recursive_quantize(module: nn.Module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(
                    module,
                    name,
                    QuantizedConv2d(
                        child, num_bits, per_channel=per_channel, pack=pack
                    ),
                )
            elif isinstance(child, nn.Linear):
                setattr(
                    module,
                    name,
                    QuantizedLinear(
                        child, num_bits, per_channel=per_channel, pack=pack
                    ),
                )
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
    """Count *in-memory* bits for an FP32 model.

    We count both parameters and buffers using their actual tensor storage
    sizes (element_size). This better matches the assignment's "when loaded in
    memory" phrasing.
    """

    total_bits = 0
    for p in model.parameters():
        total_bits += p.numel() * p.element_size() * 8
    for b in model.buffers():
        total_bits += b.numel() * b.element_size() * 8
    return int(total_bits)


def count_float_weight_bits(
    model: nn.Module, dtype_bits: int = 32
) -> int:
    """
    Count bits used by Conv2d and Linear weights only (baseline).
    """
    total_bits = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight
            total_bits += w.numel() * w.element_size() * 8
    return int(total_bits)


def count_compressed_model_bits(
    model: nn.Module, weight_bits: int
) -> Dict[str, float]:
    """
    For a quantized model using QuantizedConv2d/QuantizedLinear:
    - Computes total bits = quantized weights + scales + remaining FP32 params.
    - Also returns weight_bits_only and scale_bits_only for analysis.
    """

    # Total in-memory bits (parameters + buffers)
    total_bits = 0
    for p in model.parameters():
        total_bits += p.numel() * p.element_size() * 8
    for b in model.buffers():
        total_bits += b.numel() * b.element_size() * 8

    # Breakdowns for analysis
    quant_weight_bits = 0
    scale_bits = 0

    for module in model.modules():
        if isinstance(module, (QuantizedConv2d, QuantizedLinear)):
            # Weight storage
            if hasattr(module, "qweight_packed"):
                quant_weight_bits += module.qweight_packed.numel() * 8
            else:
                q = module.qweight
                quant_weight_bits += q.numel() * q.element_size() * 8

            # Scale storage
            s = module.scale
            scale_bits += s.numel() * s.element_size() * 8

    other_bits = total_bits - quant_weight_bits - scale_bits
    return {
        "total_bits": float(total_bits),
        "weight_bits": float(quant_weight_bits),
        "scale_bits": float(scale_bits),
        "other_bits": float(other_bits),
    }


def get_activation_compression_stats() -> Dict[str, float]:
    """
    Return activation baseline/compressed bits and compression ratio.
    """
    return default_activation_observer.summary()

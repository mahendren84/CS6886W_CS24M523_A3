import torch.nn as nn

# Handle both old and new torchvision APIs
try:
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

    _HAS_WEIGHTS_ENUM = True
except ImportError:  # older torchvision
    from torchvision.models import mobilenet_v2

    MobileNet_V2_Weights = None
    _HAS_WEIGHTS_ENUM = False


class MobileNetV2CIFAR10(nn.Module):
    """
    Wrapper around torchvision's MobileNet-v2 adapted to CIFAR-10:
    - First conv stride is set to 1 (input 32x32).
    - Classifier adjusted to 10 classes.
    """

    def __init__(
        self,
        width_mult: float = 1.0,
        dropout: float = 0.2,
        pretrained: bool = False,
        num_classes: int = 10,
    ):
        super().__init__()

        if pretrained:
            if _HAS_WEIGHTS_ENUM and MobileNet_V2_Weights is not None:
                backbone = mobilenet_v2(
                    weights=MobileNet_V2_Weights.IMAGENET1K_V1
                )
            else:
                backbone = mobilenet_v2(pretrained=True)
        else:
            if _HAS_WEIGHTS_ENUM:
                backbone = mobilenet_v2(weights=None)
            else:
                backbone = mobilenet_v2(pretrained=False)

        # Adjust width multiplier if needed
        if width_mult != 1.0:
            # Rebuild with new width multiplier if supported
            # (torchvision's mobilenet_v2 has a width_mult arg)
            if _HAS_WEIGHTS_ENUM:
                backbone = mobilenet_v2(weights=None, width_mult=width_mult)
            else:
                backbone = mobilenet_v2(
                    pretrained=False, width_mult=width_mult
                )

        # Adapt for CIFAR-10: stride 1 on first conv to keep more spatial info
        first_block = backbone.features[0]
        if hasattr(first_block, "0"):
            conv = first_block[0]
            conv.stride = (1, 1)

        # Replace classifier head for 10 classes
        in_features = backbone.classifier[1].in_features
        backbone.classifier[0].p = dropout
        backbone.classifier[1] = nn.Linear(in_features, num_classes)

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

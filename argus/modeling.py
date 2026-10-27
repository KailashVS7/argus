from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


@dataclass
class ModelBuildMetadata:
    requested_weights: str
    used_imagenet_weights: bool
    no_download: bool


class VGG16Backbone(nn.Module):
    def __init__(self, features: nn.Module) -> None:
        super().__init__()
        self.features = features
        self.out_channels = 512

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.features(x)
        return {"0": x}


def _imagenet_vgg16_cached() -> bool:
    cache_root = Path(torch.hub.get_dir()) / "checkpoints"
    filename = Path(urlparse(VGG16_Weights.IMAGENET1K_V1.url).path).name
    return (cache_root / filename).exists()


def build_model(weights: str = "none", no_download: bool = False) -> tuple[FasterRCNN, ModelBuildMetadata]:
    use_imagenet = weights == "imagenet"
    if use_imagenet and no_download and not _imagenet_vgg16_cached():
        use_imagenet = False

    vgg_weights = VGG16_Weights.IMAGENET1K_V1 if use_imagenet else None
    features = vgg16(weights=vgg_weights).features
    backbone = VGG16Backbone(features)

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=256,
        max_size=1024,
    )

    return model, ModelBuildMetadata(
        requested_weights=weights,
        used_imagenet_weights=bool(vgg_weights is not None),
        no_download=no_download,
    )


def save_checkpoint(path: str, model: nn.Module, metadata: ModelBuildMetadata) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "metadata": {
            "requested_weights": metadata.requested_weights,
            "used_imagenet_weights": metadata.used_imagenet_weights,
            "no_download": metadata.no_download,
        },
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, strict: bool = False) -> dict:
    payload = torch.load(path, map_location="cpu")
    state_dict = payload["state_dict"] if "state_dict" in payload else payload
    model.load_state_dict(state_dict, strict=strict)
    return payload.get("metadata", {})

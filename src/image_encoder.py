"""Frozen pretrained VGG-19 fc7 features averaged over ten image crops."""
from __future__ import annotations

from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms


class VGG19FC7(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        network = models.vgg19(weights=weights)
        self.features = network.features
        self.avgpool = network.avgpool
        self.fc7 = network.classifier[:6]
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.features(images)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.fc7(x)


def ten_crop_transform(image: Image.Image) -> torch.Tensor:
    resize = transforms.Resize((256, 256))
    crops = transforms.TenCrop(224)(resize(image.convert("RGB")))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return torch.stack([normalize(to_tensor(crop)) for crop in crops])

"""Model definitions used by the CompCars experiments.

The make classifier keeps the same parameter names as torchvision's ResNet-34 so
that ImageNet weights can be transferred directly. The fine-grained model uses
the same backbone and attaches one model-classification head per car make.
"""

from __future__ import annotations

from typing import Sequence

from torch import Tensor, nn
from torchvision.models.resnet import BasicBlock, ResNet


class ResNet34(ResNet):
    """ResNet-34 classifier compatible with torchvision ResNet-34 weights."""

    def __init__(self, num_classes: int) -> None:
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


class FineGrainedResNet34(nn.Module):
    """Hierarchical ResNet-34 for make followed by make-specific model labels.

    ``forward`` returns make logits and a 512-dimensional feature vector. Call
    ``classify_model(make_index, features)`` to obtain logits for the models
    belonging to the selected make.
    """

    def __init__(self, num_makes: int, model_counts: Sequence[int]) -> None:
        super().__init__()
        if len(model_counts) != num_makes:
            raise ValueError("model_counts must contain one entry for every make")
        if any(count < 1 for count in model_counts):
            raise ValueError("every make must have at least one model")

        self.resnet = ResNet34(num_classes=num_makes)
        self.resnet.fc = nn.Identity()
        self.make_classifier = nn.Linear(512, num_makes)
        self.model_classifiers = nn.ModuleList(
            nn.Linear(512, count) for count in model_counts
        )

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.resnet(images)
        make_logits = self.make_classifier(features)
        return make_logits, features

    def classify_model(self, make_index: int, features: Tensor) -> Tensor:
        """Predict the model within a make from one or more feature vectors."""
        if not 0 <= make_index < len(self.model_classifiers):
            raise IndexError(f"Unknown make index: {make_index}")
        return self.model_classifiers[make_index](features)

"""
DR severity classifier: EfficientNet-B0 backbone (pretrained on ImageNet)
with a dropout + linear head for 5-class severity grading.

EfficientNet-B0 is chosen over ResNet50 for a lighter, faster model that
still performs competitively on fundus classification (per the literature
review, EfficientNet variants are consistently among the top performers).
"""
import torch
import torch.nn as nn
import torchvision.models as models


class DRClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.35, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Keep the conv feature extractor; replace the classifier head.
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        in_features = backbone.classifier[1].in_features

        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.features(x)
        pooled = self.avgpool(feats).flatten(1)
        out = self.classifier(self.dropout(pooled))
        return out

    def forward_with_features(self, x):
        """Returns both logits and the last conv feature map (for Grad-CAM)."""
        feats = self.features(x)
        pooled = self.avgpool(feats).flatten(1)
        out = self.classifier(self.dropout(pooled))
        return out, feats


def build_model(num_classes: int = 5, device: str = "cuda") -> DRClassifier:
    model = DRClassifier(num_classes=num_classes)
    return model.to(device)

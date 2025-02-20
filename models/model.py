# models/model.py

from .backbones import get_backbone
from torch import nn


class CustomModel(nn.Module):
    def __init__(self, model_name, use_weights=True, num_classes=1000):
        super().__init__()
        self.backbone = get_backbone(model_name, use_weights)
        self.fc = nn.Linear(self.backbone.out_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)

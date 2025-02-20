# models/backbones.py

import torchvision.models as models
from torch import nn


class BackboneWithoutClassifier(nn.Module):
    def __init__(self, backbone, out_features):
        super().__init__()
        self.backbone = backbone
        self.out_features = out_features

    def forward(self, x):
        return self.backbone(x)


def get_backbone(model_name, use_weights=True):
    weights = "DEFAULT" if use_weights else None

    backbones = {
        "resnet50": models.resnet50,
        "vgg16": models.vgg16,
        "mobilenet_v2": models.mobilenet_v2,
        "inception_v3": models.inception_v3,
        "convnext_tiny": models.convnext_tiny,
        "resnext50_32x4d": models.resnext50_32x4d,
        "vit_b_16": models.vit_b_16,
        "vit_l_32": models.vit_l_32,
        "swin_v2_b": models.swin_v2_b,
        "swin_v2_t": models.swin_v2_t,
    }

    if model_name not in backbones:
        raise ValueError(f"Model {model_name} not supported.")

    backbone = backbones[model_name](weights=weights)

    # Remove the classifier and get the output features
    if hasattr(backbone, "fc"):
        out_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        if isinstance(backbone.classifier, nn.Sequential):
            out_features = backbone.classifier[-1].in_features
        else:
            out_features = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    elif hasattr(backbone, "heads"):
        out_features = backbone.heads.head.in_features
        backbone.heads.head = nn.Identity()
    elif hasattr(backbone, "head"):
        out_features = backbone.head.in_features
        backbone.head = nn.Identity()
    else:
        raise ValueError(f"Unsupported backbone structure: {model_name}")

    if use_weights:
        # Freeze all layers initially
        for param in backbone.parameters():
            param.requires_grad = False

        # Find the last two trainable layers
        trainable_layers = []
        for module in backbone.modules():
            if any(param.requires_grad for param in module.parameters()):
                trainable_layers.append(module)

        # Unfreeze the last two trainable layers
        for layer in trainable_layers[-4:]:
            for param in layer.parameters():
                param.requires_grad = True

    return BackboneWithoutClassifier(backbone, out_features)

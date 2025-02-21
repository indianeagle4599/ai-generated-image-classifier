# models/model.py

from .backbones import get_backbone
from torch import nn

import torch


class CustomModel(nn.Module):
    def __init__(self, model_name="", use_weights=True, num_classes=1000, features=[]):
        super().__init__()
        self.total_features = 0
        self.has_backbone = False
        self.features = features
        self.feature_processors = []

        if model_name:
            self.backbone = get_backbone(model_name, use_weights)
            self.backbone_bn_1 = nn.BatchNorm1d(self.backbone.out_features)
            self.backbone_fc_1 = nn.Linear(self.backbone.out_features, 512)
            self.backbone_relu_1 = nn.ReLU()
            self.backbone_dropout_1 = nn.Dropout(0.4)

            self.backbone_procesor = nn.Sequential(
                self.backbone,
                self.backbone_bn_1,
                self.backbone_fc_1,
                self.backbone_relu_1,
                self.backbone_dropout_1,
            )

            self.has_backbone = True
            self.total_features += 512

        if "FFT" in features:
            fft_cnn_input_layer = nn.Conv2d(3, 32, kernel_size=5)
            fft_cnn_layer_1 = nn.Conv2d(32, 64, kernel_size=5)
            fft_max_pool_1 = nn.MaxPool2d(2, 2)
            fft_cnn_layer_2 = nn.Conv2d(64, 128, kernel_size=3)
            fft_cnn_layer_3 = nn.Conv2d(128, 128, kernel_size=3)
            fft_adaptive_pool = nn.AdaptiveAvgPool2d(4)
            fft_flatten = nn.Flatten()
            fft_bn_1 = nn.BatchNorm1d(2048)
            fft_fc_1 = nn.Linear(2048, 512)
            fft_relu_1 = nn.ReLU()
            fft_dropout_1 = nn.Dropout(0.4)

            self.fft_processor = nn.Sequential(
                fft_cnn_input_layer,
                fft_cnn_layer_1,
                fft_max_pool_1,
                fft_cnn_layer_2,
                fft_cnn_layer_3,
                fft_adaptive_pool,
                fft_flatten,
                fft_bn_1,
                fft_fc_1,
                fft_relu_1,
                fft_dropout_1,
            )

            self.feature_processors.append(self.fft_processor)
            self.total_features += 512

        if "DCT" in features:
            dct_bn_1 = nn.BatchNorm1d(64)
            dct_fc_1 = nn.Linear(64, 512)
            dct_relu_1 = nn.ReLU()
            dct_dropout_1 = nn.Dropout(0.4)

            self.dct_processor = nn.Sequential(
                dct_bn_1,
                dct_fc_1,
                dct_relu_1,
                dct_dropout_1,
            )

            self.feature_processors.append(self.dct_processor)
            self.total_features += 512

        classifier_fc_1 = nn.Linear(self.total_features, self.total_features)
        classifier_relu_1 = nn.ReLU()
        classifier_dropout_1 = nn.Dropout(0.4)
        classifier_bn_1 = nn.BatchNorm1d(self.total_features)
        classifier_fc_2 = nn.Linear(self.total_features, 512)
        classifier_relu_2 = nn.ReLU()
        classifier_dropout_2 = nn.Dropout(0.4)
        classifier_fc_3 = nn.Linear(512, num_classes)

        self.classifier = nn.Sequential(
            classifier_fc_1,
            classifier_relu_1,
            classifier_dropout_1,
            classifier_bn_1,
            classifier_fc_2,
            classifier_relu_2,
            classifier_dropout_2,
            classifier_fc_3,
        )

    def forward(self, x, device):
        processed_features = torch.Tensor().to(device)

        if self.features:
            x, features = x
            x = x.to(device)
            for i, feature in enumerate(features):
                extra_features = self.feature_processors[i](feature.to(device))
                processed_features = torch.concat(
                    [processed_features, extra_features], dim=1
                )

        if self.has_backbone:
            backbone_features = self.backbone_procesor(x)
            processed_features = torch.concat([processed_features, backbone_features])

        x = self.classifier(processed_features)
        return x

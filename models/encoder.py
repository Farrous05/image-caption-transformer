"""CNN Encoder — uses pretrained ResNet-50 to extract spatial features."""

import torch
import torch.nn as nn
import torchvision.models as models

import config


class Encoder(nn.Module):
    """
    Takes an image, runs it through ResNet-50 (without the final pooling/fc),
    and projects the 7x7 feature grid to embed_dim.
    Output shape: (batch, 49, embed_dim)
    """

    def __init__(self, embed_dim=config.EMBED_DIM, fine_tune=False):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # drop avgpool + fc, keep everything up to last conv block
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # 2048 -> embed_dim
        self.projection = nn.Linear(config.ENCODER_DIM, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)

        self._set_fine_tune(fine_tune)

    def _set_fine_tune(self, fine_tune):
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune

    def forward(self, images):
        # (batch, 2048, 7, 7)
        features = self.resnet(images)
        batch_size = features.size(0)

        # reshape to (batch, 49, 2048) then project
        features = features.view(batch_size, config.ENCODER_DIM, -1)
        features = features.permute(0, 2, 1)
        features = self.dropout(self.relu(self.projection(features)))

        return features

"""
CNN Encoder using pre-trained ResNet-50.

Extracts spatial feature maps from images and projects them
to the model's embedding dimension.

Output: (batch_size, 49, embed_dim) — 49 spatial patches from the 7x7 feature grid.
"""

import torch
import torch.nn as nn
import torchvision.models as models

import config


class Encoder(nn.Module):
    """
    ResNet-50 based image encoder.

    Takes an image and produces a sequence of spatial feature vectors
    that the Transformer decoder can attend to.
    """

    def __init__(self, embed_dim=config.EMBED_DIM, fine_tune=False):
        super().__init__()

        # Load pre-trained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Remove the final avgpool and fc layers
        # We want the feature maps from the last conv block
        modules = list(resnet.children())[:-2]  # Remove avgpool + fc
        self.resnet = nn.Sequential(*modules)

        # Project from ResNet feature dim (2048) to model embed dim
        self.projection = nn.Linear(config.ENCODER_DIM, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.DROPOUT)

        # Freeze/unfreeze ResNet parameters
        self._set_fine_tune(fine_tune)

    def _set_fine_tune(self, fine_tune):
        """Enable or disable gradient computation for ResNet layers."""
        for param in self.resnet.parameters():
            param.requires_grad = fine_tune

    def forward(self, images):
        """
        Args:
            images: (batch_size, 3, 224, 224)

        Returns:
            features: (batch_size, 49, embed_dim)
                49 = 7*7 spatial patches, each projected to embed_dim
        """
        # Extract feature maps: (batch, 2048, 7, 7)
        features = self.resnet(images)

        batch_size = features.size(0)

        # Reshape: (batch, 2048, 7, 7) → (batch, 2048, 49) → (batch, 49, 2048)
        features = features.view(batch_size, config.ENCODER_DIM, -1)
        features = features.permute(0, 2, 1)

        # Project to embed_dim: (batch, 49, 2048) → (batch, 49, embed_dim)
        features = self.dropout(self.relu(self.projection(features)))

        return features

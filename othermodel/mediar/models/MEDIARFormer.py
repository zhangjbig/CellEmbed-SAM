import torch
import torch.nn as nn

from segmentation_models_pytorch import MAnet
from segmentation_models_pytorch.base.modules import Activation

__all__ = ["MEDIARFormer"]


class MEDIARFormer(MAnet):
    """MEDIAR-Former Model"""

    def __init__(
        self,
        encoder_name="mit_b5",
        encoder_weights="imagenet",
        decoder_channels=(1024, 512, 256, 128, 64),
        decoder_pab_channels=256,
        in_channels=3,
        classes=3,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_channels=decoder_channels,
            decoder_pab_channels=decoder_pab_channels,
            in_channels=in_channels,
            classes=classes,
        )

        self.segmentation_head = None

        _convert_activations(self.encoder, nn.ReLU, nn.Mish(inplace=True))
        _convert_activations(self.decoder, nn.ReLU, nn.Mish(inplace=True))

        self.cellprob_head = DeepSegmentationHead(
            in_channels=decoder_channels[-1], out_channels=1
        )
        self.gradflow_head = DeepSegmentationHead(
            in_channels=decoder_channels[-1], out_channels=2
        )

    def forward(self, x):
        """Forward pass through the network"""
        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(features)

        cellprob_mask = self.cellprob_head(decoder_output)
        gradflow_mask = self.gradflow_head(decoder_output)

        masks = torch.cat([gradflow_mask, cellprob_mask], dim=1)

        return masks


class DeepSegmentationHead(nn.Sequential):
    """Custom segmentation head for generating specific masks"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1
    ):
        layers = [
            nn.Conv2d(
                in_channels,
                in_channels // 2,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.Mish(inplace=True),
            nn.BatchNorm2d(in_channels // 2),
            nn.Conv2d(
                in_channels // 2,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ),
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity(),
            Activation(activation) if activation else nn.Identity(),
        ]
        super().__init__(*layers)


def _convert_activations(module, from_activation, to_activation):
    """Recursively convert activation functions in a module"""
    for name, child in module.named_children():
        if isinstance(child, from_activation):
            setattr(module, name, to_activation)
        else:
            _convert_activations(child, from_activation, to_activation)

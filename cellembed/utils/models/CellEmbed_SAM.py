from collections import OrderedDict
from einops import rearrange
from functools import partial
from pathlib import Path
from typing import List, Literal, Tuple, Union, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cellembed.utils.models.SAM.utils import ViTCellViT
from cellembed.utils.models.SAM.image_encoder import ImageEncoderViT
from cellembed.utils.models.SAM.post_proc_cellvit import DetectionCellPostProcessor


def conv_norm_act(in_channels, out_channels, sz, norm, act="ReLU", depthwise=False):
    if norm == "None" or norm is None:
        norm_layer = nn.Identity()
    elif norm.lower() == "batch":
        norm_layer = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.05)
    elif norm.lower() == "instance":
        norm_layer = nn.InstanceNorm2d(out_channels, eps=1e-5, track_running_stats=False, affine=True)
    else:
        raise ValueError("Norm must be None, batch or instance")

    if act == "None" or act is None:
        act_layer = nn.Identity()
    elif act.lower() == "relu":
        act_layer = nn.ReLU(inplace=True)
    elif act.lower() == "relu6":
        act_layer = nn.ReLU6(inplace=True)
    elif act.lower() == "mish":
        act_layer = nn.Mish(inplace=True)
    else:
        raise ValueError("Act must be None, ReLU or Mish")

    if depthwise:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2, groups=in_channels),
            norm_layer,
            act_layer,
        )
    else:

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, sz, padding=sz // 2),
            norm_layer,
            act_layer,
        )



class Conv2DBlock(nn.Module):
    """Conv2DBlock: convolution followed by normalisation, activation, and optional dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size. Default: 3.
        dropout (float, optional): Dropout probability. Default: 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()

        norm = "BATCH"
        act = "ReLU"

        self.conv0 = conv_norm_act(in_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(in_channels, out_channels, kernel_size, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, kernel_size, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, kernel_size, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, kernel_size, norm, act)

    def forward(self, x):

        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x)
        x = x + self.conv4(self.conv3(x))

        return x

class Deconv2DBlock(nn.Module):
    """Deconvolution block: ConvTranspose2d followed by Conv2d, normalisation, ReLU activation, and dropout.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size. Default: 3.
        dropout (float, optional): Dropout probability. Default: 0.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=((kernel_size - 1) // 2),
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):

    """Decoder block with skip connections and stacked convolutional layers."""

    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            norm="BATCH",
            act="ReLU",
            shallow=False,
    ):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv0 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv_skip = conv_norm_act(skip_channels, out_channels, 1, norm, act)
        self.conv1 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv2 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv3 = conv_norm_act(out_channels, out_channels, 3, norm, act)
        self.conv4 = conv_norm_act(out_channels, out_channels, 3, norm, act)

        if shallow:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=None):

        x = self.conv_transpose(x)
        proj = self.conv0(x)
        x = self.conv1(x)
        x = proj + self.conv2(x + self.conv_skip(skip))
        x = x + self.conv4(self.conv3(x))

        return x


class Decoder(nn.Module):

    """Full decoder consisting of multiple DecoderBlocks and a final output block."""

    def __init__(
            self,
            layers,
            out_channels,
            norm,
            act):
        super().__init__()

        if isinstance(out_channels, (list, tuple)):
            out_channels = [int(c[0] if isinstance(c, (list, tuple)) else c) for c in out_channels]
        else:
            out_channels = [int(out_channels)]
        assert all(isinstance(c, int) for c in out_channels)


        self.decoder = nn.ModuleList([DecoderBlock(layers[i], layers[i + 1], layers[i + 1], norm=norm, act=act) for i in range(len(layers) - 1)])
        self.final_block = nn.ModuleList([conv_norm_act(layers[-1], out_channel, 1, norm=norm if (norm is not None) and norm.lower() != "instance" else None, act=None) for out_channel in out_channels])

    def forward(self, x, skips):
        for layer, skip in zip(self.decoder, skips[::-1]):
            x = layer(x, skip)

        x = torch.cat([final_block(x) for final_block in self.final_block], dim=1)

        return x


class EdgeEnhanceRefiner(nn.Module):
    """Edge enhancement module supporting Sobel / DoG filtering.
    Non-trainable, designed for refining cell structures."""

    def __init__(self, in_channels=1, noise_std=0.003, edge_mode='DoG'):
        """
        Args:
            in_channels (int): Number of input channels.
            noise_std (float): Std of additive Gaussian noise during training.
            edge_mode (str): 'Sobel' or 'DoG'. Determines edge enhancement filter.
        """
        super().__init__()
        self.noise_std = noise_std
        self.edge_mode = edge_mode

        if edge_mode == 'Sobel':
            # Sobel: directional edge detection
            sobel_x = torch.tensor([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]], dtype=torch.float32)
            sobel_y = torch.tensor([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]], dtype=torch.float32)
            self.sobel_x = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False, groups=in_channels)
            self.sobel_y = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False, groups=in_channels)
            self.sobel_x.weight.data = sobel_x.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
            self.sobel_y.weight.data = sobel_y.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)
            self.sobel_x.weight.requires_grad = False
            self.sobel_y.weight.requires_grad = False

        elif edge_mode == 'DoG':
            # Difference of Gaussians: Gaussian(σ1) - Gaussian(σ2), simulates a high-pass filter
            self.gaussian1 = nn.Conv2d(in_channels, in_channels, 5, 1, 2, bias=False, groups=in_channels)
            self.gaussian2 = nn.Conv2d(in_channels, in_channels, 5, 1, 2, bias=False, groups=in_channels)
            self._init_gaussian(self.gaussian1, sigma=1.0)
            self._init_gaussian(self.gaussian2, sigma=2.0)
            self.gaussian1.weight.requires_grad = False
            self.gaussian2.weight.requires_grad = False

    def _init_gaussian(self, conv, sigma):
        """Initialise convolution weights with Gaussian kernel."""
        ksize = conv.kernel_size[0]
        ax = torch.arange(-ksize // 2 + 1., ksize // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
        kernel = kernel / kernel.sum()
        conv.weight.data = kernel.view(1, 1, ksize, ksize).repeat(conv.in_channels, 1, 1, 1)

    def forward(self, x):
        # Upsample ×2
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Add Gaussian noise during training
        if self.training:
            x = x + self.noise_std * torch.randn_like(x)

        # Edge enhancement
        if self.edge_mode == 'Sobel':
            gx = self.sobel_x(x)
            gy = self.sobel_y(x)
            edge = torch.sqrt(gx ** 2 + gy ** 2)
            x = x + 0.1 * edge

        elif self.edge_mode == 'DoG':
            blur1 = self.gaussian1(x)
            blur2 = self.gaussian2(x)
            edge = blur1 - blur2
            x = x + 0.1 * edge

        return x


class CellEmbed_Encoder(ImageEncoderViT):
    def __init__(
        self,
        extract_layers: List[int],
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            out_chans,
            qkv_bias,
            norm_layer,
            act_layer,
            use_abs_pos,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            global_attn_indexes,
        )
        self.extract_layers = extract_layers
        self.ps = 8
        self.fine = EdgeEnhanceRefiner(in_channels=3, edge_mode="DoG")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        extracted_layers = []

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # b c h w

        scale_factor = self.patch_embed.proj.kernel_size[0] / self.ps  # e.g. 16/8 = 2

        if scale_factor != 1.0:
            x = self.fine(x)

        x = self.patch_embed(x)  # b h w c

        if self.pos_embed is not None:

            token_size1 = x.shape[1]
            token_size2 = x.shape[2]

            x = x + self.pos_embed[:, :token_size1, :token_size2, :]

        for depth, blk in enumerate(self.blocks):
            x = blk(x)

            if depth + 1 in self.extract_layers:
                extracted_layers.append(x)

        output = self.neck(extracted_layers[-1].permute(0, 3, 1, 2))
        _output = rearrange(output, "b c h w -> b c (h w)")

        return torch.mean(_output, axis=-1), extracted_layers[-1].permute(0, 3, 1, 2), extracted_layers


class CellViT_Instanseg(nn.Module):
    def __init__(
            self,
            num_nuclei_classes: int,      # Number of nucleus classes (including background)
            num_tissue_classes: int,      # Number of tissue classes
            embed_dim: int,               # ViT embedding dimension (token feature size)
            input_channels: int,          # Number of input image channels
            depth: int,                   # Number of Transformer blocks in ViT
            num_heads: int,               # Number of attention heads in ViT
            extract_layers: List,         # Transformer block indices used for skip connections
            mlp_ratio: float = 4,         # Hidden dimension ratio in MLP block
            qkv_bias: bool = True,        # Whether to use bias in Q/K/V projections
            drop_rate: float = 0,         # Dropout rate in MLP
            attn_drop_rate: float = 0,    # Dropout rate in attention
            drop_path_rate: float = 0,    # Stochastic depth rate (DropPath)
            regression_loss: bool = False # Whether to use regression loss
    ):
        # For simplicity, we assume exactly 4 layers are used for skip connections
        super().__init__()
        assert len(extract_layers) == 4, "Please provide 4 layers for skip connections"

        self.patch_size = 16
        self.num_tissue_classes = num_tissue_classes
        self.num_nuclei_classes = num_nuclei_classes
        self.embed_dim = embed_dim
        self.input_channels = input_channels
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.extract_layers = extract_layers
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        # ViT encoder
        self.encoder = ViTCellViT(
            patch_size=self.patch_size,                  # Patch size for tokenization
            num_classes=self.num_tissue_classes,         # Number of tissue classes
            embed_dim=self.embed_dim,                    # ViT embedding dimension
            depth=self.depth,                            # Number of Transformer layers
            num_heads=self.num_heads,                    # Number of attention heads
            mlp_ratio=self.mlp_ratio,                    # MLP hidden size ratio
            qkv_bias=self.qkv_bias,                      # Use bias in Q/K/V
            norm_layer=partial(nn.LayerNorm, eps=1e-6),  # Normalization layer
            extract_layers=self.extract_layers,          # Layers to extract for skip connections
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Channel configuration for skip branches depending on encoder width
        if self.embed_dim < 512:
            self.skip_dim_11 = 256
            self.skip_dim_12 = 128
            self.bottleneck_dim = 312
        else:
            self.skip_dim_11 = 512
            self.skip_dim_12 = 256
            self.bottleneck_dim = 512

        # Version with shared skip connections
        self.decoder0 = nn.Sequential(
            Conv2DBlock(3, 32, 3, dropout=self.drop_rate),
            Conv2DBlock(32, 64, 3, dropout=self.drop_rate),
        )  # Skip after positional encoding; expected spatial shape H x W with 64 channels

        self.decoder1 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, self.skip_dim_12, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_12, 128, dropout=self.drop_rate),
        )  # Skip connection 1

        self.decoder2 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.skip_dim_11, dropout=self.drop_rate),
            Deconv2DBlock(self.skip_dim_11, 256, dropout=self.drop_rate),
        )  # Skip connection 2

        self.decoder3 = nn.Sequential(
            Deconv2DBlock(self.embed_dim, self.bottleneck_dim, dropout=self.drop_rate)
        )  # Skip connection 3

        # Channel plan across decoder stages (from deep to shallow)
        self.layers = [384, 312, 256, 128, 64]

        norm = "BATCH"
        act = "ReLu"
        # Three prediction heads concatenated along channel dim (e.g., [2], [2], [1])
        out_channels = [[2], [2], [1]]
        self.decoders = nn.ModuleList(
            [Decoder(self.layers, out_channel, norm, act) for out_channel in out_channels]
        )

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False) -> dict:
        # Ensure spatial dimensions are divisible by the patch size
        assert x.shape[-2] % self.patch_size == 0, "Input height must be divisible by patch_size"
        assert x.shape[-1] % self.patch_size == 0, "Input width must be divisible by patch_size"

        # Encoder forward pass
        classifier_logits, _, z = self.encoder(x)

        # Unpack skip tokens (z1..z4) together with raw input (z0)
        z0, z1, z2, z3, z4 = x, *z

        # Reshape tokens back to feature maps for convolutional decoding (restore spatial dims)
        patch_dim = [int(d / self.patch_size) for d in [x.shape[-2], x.shape[-1]]]
        z4 = z4[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim).contiguous()
        z3 = z3[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim).contiguous()
        z2 = z2[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim).contiguous()
        z1 = z1[:, 1:, :].transpose(-1, -2).view(-1, self.embed_dim, *patch_dim).contiguous()

        # Multi-scale decoder branches
        b3 = self.decoder3(z3.contiguous()).contiguous()  # B x 312 x 32 x 32
        b2 = self.decoder2(z2.contiguous()).contiguous()  # B x 256 x 64 x 64
        b1 = self.decoder1(z1.contiguous()).contiguous()  # B x 128 x 128 x 128
        b0 = self.decoder0(z0.contiguous()).contiguous()  # B x 64  x 256 x 256

        # Assemble skip list from shallow to deep
        skips = [b0, b1, b2, b3]

        # Run all task decoders and concatenate their outputs along channel dimension
        out_dict = torch.cat([decoder(z4, skips) for decoder in self.decoders], dim=1)
        return out_dict

    def calculate_instance_map(
        self, predictions: OrderedDict, magnification: Literal[20, 40] = 40
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Compute instance map from network predictions (after softmax).

        Args:
            predictions (dict): Must contain:
                * nuclei_binary_map: Binary nucleus predictions. Shape: (B, 2, H, W)
                * nuclei_type_map: Nucleus type logits or probs. Shape: (B, self.num_nuclei_classes, H, W)
                * hv_map: Horizontal-vertical map. Shape: (B, 2, H, W)
            magnification (Literal[20, 40]): Data magnification used. Default: 40.

        Returns:
            Tuple[torch.Tensor, List[dict]]:
                * Tensor: Instance map with unique integer per instance. Shape: (B, H, W)
                * List[dict]: For each image, a dict of nuclei properties:
                    "bbox", "centroid", "contour", "type_prob", "type"
        """
        # Reorder to (B, H, W, C)
        predictions_ = predictions.copy()
        predictions_["nuclei_type_map"] = predictions_["nuclei_type_map"].permute(0, 2, 3, 1)
        predictions_["nuclei_binary_map"] = predictions_["nuclei_binary_map"].permute(0, 2, 3, 1)
        predictions_["hv_map"] = predictions_["hv_map"].permute(0, 2, 3, 1)

        cell_post_processor = DetectionCellPostProcessor(
            nr_types=self.num_nuclei_classes, magnification=magnification, gt=False
        )
        instance_preds = []
        type_preds = []

        for i in range(predictions_["nuclei_binary_map"].shape[0]):
            pred_map = np.concatenate(
                [
                    torch.argmax(predictions_["nuclei_type_map"], dim=-1)[i].detach().cpu()[..., None],
                    torch.argmax(predictions_["nuclei_binary_map"], dim=-1)[i].detach().cpu()[..., None],
                    predictions_["hv_map"][i].detach().cpu(),
                ],
                axis=-1,
            )
            instance_pred = cell_post_processor.post_process_cell_segmentation(pred_map)
            instance_preds.append(instance_pred[0])
            type_preds.append(instance_pred[1])

        return torch.Tensor(np.stack(instance_preds)), type_preds

    def generate_instance_nuclei_map(
        self, instance_maps: torch.Tensor, type_preds: List[dict]
    ) -> torch.Tensor:
        """Convert binary instance map to a nuclei-type instance map.

        Args:
            instance_maps (torch.Tensor): Instance map with unique ids per object. Shape: (B, H, W)
            type_preds (List[dict]): Per-image dict of instance type info
                                     (see post_process_hovernet for details).

        Returns:
            torch.Tensor: Nuclei-type instance map. Shape: (B, self.num_nuclei_classes, H, W)
        """
        batch_size, h, w = instance_maps.shape
        instance_type_nuclei_maps = torch.zeros((batch_size, h, w, self.num_nuclei_classes))
        for i in range(batch_size):
            instance_type_nuclei_map = torch.zeros((h, w, self.num_nuclei_classes))
            instance_map = instance_maps[i]
            type_pred = type_preds[i]
            for nuclei, spec in type_pred.items():
                nuclei_type = spec["type"]
                instance_type_nuclei_map[:, :, nuclei_type][instance_map == nuclei] = nuclei

            instance_type_nuclei_maps[i, :, :, :] = instance_type_nuclei_map

        instance_type_nuclei_maps = instance_type_nuclei_maps.permute(0, 3, 1, 2)
        return torch.Tensor(instance_type_nuclei_maps)

    def freeze_encoder(self):
        """Freeze encoder parameters except the classification head."""
        for layer_name, p in self.encoder.named_parameters():
            if layer_name.split(".")[0] != "head":  # do not freeze the head
                p.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True



class CellEmbed_SAM(CellViT_Instanseg):

    """CellEmbed with a SAM backbone.

    Skip connections are shared across branches, while each task head has its own decoder.
    Each network uses the same encoder interface but may load distinct checkpoints.

    Args:
        model_path (Union[Path, str]): Path to a pretrained SAM checkpoint.
        num_nuclei_classes (int): Number of nucleus classes (including background).
        num_tissue_classes (int): Number of tissue classes.
        vit_structure (Literal["SAM-B", "SAM-L", "SAM-H"]): SAM backbone size.
        drop_rate (float, optional): Dropout rate used for MLP/blocks. Default: 0.
        regression_loss (bool, optional): If True, use a regressive loss for vector components;
            this adds two extra channels to the binary decoder, returned as a separate entry. Default: False.
        out_channels (list, optional): Per-head output channel specification (e.g., [[2],[2],[1]]).
        layers (list, optional): Channel configuration used by the decoders (deep->shallow).

    Raises:
        NotImplementedError: If an unknown SAM configuration is requested.
    """

    def __init__(
        self,
        model_path: Union[Path, str],
        num_nuclei_classes: int,
        num_tissue_classes: int,
        vit_structure: Literal["SAM-B", "SAM-L", "SAM-H"],
        drop_rate: float = 0,
        regression_loss: bool = False,
        out_channels: list = [],
        layers: list = [],
    ):
        # Select backbone configuration by size
        if vit_structure.upper() == "SAM-B":
            self.init_vit_b()
        elif vit_structure.upper() == "SAM-L":
            self.init_vit_l()
        elif vit_structure.upper() == "SAM-H":
            self.init_vit_h()
        else:
            raise NotImplementedError("Unknown ViT-SAM backbone structure")

        self.input_channels = 3  # RGB input
        self.mlp_ratio = 4
        self.qkv_bias = True
        self.num_nuclei_classes = num_nuclei_classes
        self.model_path = model_path

        # Base InstanSeg-style ViT encoder + multi-branch decoder scaffold
        super().__init__(
            num_nuclei_classes=num_nuclei_classes,
            num_tissue_classes=num_tissue_classes,
            embed_dim=self.embed_dim,
            input_channels=self.input_channels,
            depth=self.depth,
            num_heads=self.num_heads,
            extract_layers=self.extract_layers,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=drop_rate,
            drop_path_rate=drop_rate,
            regression_loss=regression_loss,
        )

        # Replace the encoder with a SAM-style image encoder producing prompt embeddings
        self.prompt_embed_dim = 256
        self.encoder = CellEmbed_Encoder(
            extract_layers=self.extract_layers,
            depth=self.depth,
            embed_dim=self.embed_dim,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=self.num_heads,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=self.encoder_global_attn_indexes,
            window_size=14,
            out_chans=self.prompt_embed_dim,
        )

        # Force global attention in all blocks (disable windowed attention)
        for blk in self.encoder.blocks:
            blk.window_size = 0

        # Decoder channel plan (deep -> shallow) comes from args.layers
        # layers is expected e.g. [C3, C2, C1], then we prepend embed_dim for [C4, C3, C2, C1]
        self.layers = [self.embed_dim, layers[2], layers[1], layers[0]]

        # 1x1 projections to align skip features from intermediate encoder scales
        self.cnn_out1 = nn.Conv2d(self.embed_dim, self.layers[3], kernel_size=1)
        self.cnn_out2 = nn.Conv2d(self.embed_dim, self.layers[2], kernel_size=1)
        self.cnn_out3 = nn.Conv2d(self.embed_dim, self.layers[1], kernel_size=1)

        norm = "BATCH"
        act = "ReLu"

        # Normalize out_channels argument into a list of lists (one list per head)
        if isinstance(out_channels, int):
            out_channels = [[out_channels]]
        if isinstance(out_channels[0], int):
            out_channels = [out_channels]

        # Build task decoders; each Decoder returns its head output, later concatenated on channel dim
        self.decoders = nn.ModuleList(
            [Decoder(self.layers, out_channel, norm, act) for out_channel in out_channels]
        )

        # Remove the parent decoders that are not used in the SAM variant
        del self.decoder0
        del self.decoder1
        del self.decoder2
        del self.decoder3

    def load_pretrained_encoder(self, model_path):
        """Load a pretrained SAM encoder checkpoint.

        Args:
            model_path (str or Path): Path to the SAM checkpoint file.
        """
        state_dict = torch.load(str(model_path), map_location="cpu")
        image_encoder = self.encoder
        msg = image_encoder.load_state_dict(state_dict, strict=False)
        print(f"Loading checkpoint: {msg}")
        self.encoder = image_encoder

    def forward(self, x: torch.Tensor, retrieve_tokens: bool = False):
        # Ensure spatial sizes are divisible by the patch size (token size)
        assert x.shape[-2] % self.patch_size == 0, "Image height must be divisible by patch_size (token_size)"
        assert x.shape[-1] % self.patch_size == 0, "Image width must be divisible by patch_size (token_size)"

        # Encode and collect multi-scale tokens
        classifier_logits, _, z = self.encoder(x)
        z0, z1, z2, z3, z4 = x, *z

        # Reshape encoder outputs to NCHW for convolutional decoders
        z4 = z4.permute(0, 3, 1, 2).contiguous()
        z3 = z3.permute(0, 3, 1, 2).contiguous()
        z2 = z2.permute(0, 3, 1, 2).contiguous()
        z1 = z1.permute(0, 3, 1, 2).contiguous()

        # Build skip features by upsampling intermediate scales to the shallowest resolution
        b1 = self.cnn_out1(F.interpolate(z1, scale_factor=8, mode='bilinear', align_corners=False))
        b2 = self.cnn_out2(F.interpolate(z2, scale_factor=4, mode='bilinear', align_corners=False))
        b3 = self.cnn_out3(F.interpolate(z3, scale_factor=2, mode='bilinear', align_corners=False))

        skips = [b1, b2, b3]

        # Run all task decoders and concatenate their outputs along channel dimension
        out_dict = torch.cat([decoder(z4, skips) for decoder in self.decoders], dim=1)
        return out_dict

    def init_vit_b(self):
        """Initialize SAM-B (ViT-B) backbone configuration."""
        self.embed_dim = 768
        self.depth = 12
        self.num_heads = 12
        self.interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]
        self.encoder_global_attn_indexes = [2, 5, 8, 11]
        self.extract_layers = [3, 6, 9, 12]

    def init_vit_l(self):
        """Initialize SAM-L (ViT-L) backbone configuration."""
        self.embed_dim = 1024
        self.depth = 24
        self.num_heads = 16
        self.interaction_indexes = [[0, 5], [6, 11], [12, 17], [18, 23]]
        self.encoder_global_attn_indexes = [5, 11, 17, 23]
        self.extract_layers = [6, 12, 18, 24]

    def init_vit_h(self):
        """Initialize SAM-H (ViT-H) backbone configuration."""
        self.embed_dim = 1280
        self.depth = 32
        self.num_heads = 16
        self.interaction_indexes = [[0, 7], [8, 15], [16, 23], [24, 31]]
        self.encoder_global_attn_indexes = [7, 15, 23, 31]
        self.extract_layers = [8, 16, 24, 32]

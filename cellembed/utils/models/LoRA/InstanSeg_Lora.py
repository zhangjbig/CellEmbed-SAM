import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from cellembed.utils.models.CellEmbed_SAM import CellViT_Instanseg

# _LoRA_qkv: Apply LoRA (low-rank adaptation) on the qkv projection.
class _LoRA_qkv(nn.Module):
    r"""Reference (typical ViT style, for context only):
    In SAM it is implemented as:
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

    NOTE:
    - In a standard ViT block, qkv is (B, N, 3*dim) before reshaping.
    - In this codebase, the SAM branch uses a 4D layout for qkv (see indexing below).
      We keep the implementation as-is and document both layouts.
    """

    """
    Attributes:
        qkv (nn.Module): The base linear projection producing Q, K, and V.
        linear_a_q (nn.Module): LoRA A matrix for Q branch.
        linear_b_q (nn.Module): LoRA B matrix for Q branch.
        linear_a_v (nn.Module): LoRA A matrix for V branch.
        linear_b_v (nn.Module): LoRA B matrix for V branch.
        lora_dropout (float or callable): Dropout applied to LoRA-augmented outputs.
        dim (int): Input feature dimension (= qkv.in_features).
        w_identity (torch.Tensor): Identity (not used for computation here, kept for reference).
    """

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        # linear_a_k: nn.Module,  # If LoRA for K is needed, enable these and add below.
        # linear_b_k: nn.Module,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        # self.linear_a_k = linear_a_k
        # self.linear_b_k = linear_b_k

        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            # no-op when dropout rate is 0
            self.lora_dropout = lambda x: x

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, N, C), where
                B: batch size, N: number of tokens, C: feature dimension.

        Returns:
            Tensor: qkv tensor with LoRA increments applied on q and v parts.
                    NOTE: In this code path, qkv is treated as a 4D tensor,
                    consistent with the SAM branch used in this repository.
        """

        # In standard ViT: qkv = self.qkv(x) -> (B, N, 3*dim)
        # Here (SAM branch): qkv is handled as 4D; keep as-is per original implementation.
        qkv = self.qkv(x)  # Comment in original code says: B, N, N, 3*org_C

        # LoRA increments for Q and V (A then B: low-rank factorization)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # If LoRA for K is needed, enable the two lines below and add to the slice:
        # new_k = self.linear_b_k(self.linear_a_v(x))

        # Apply LoRA increments to the qkv tensor.
        # NOTE: Indexing assumes a 4D qkv layout in this codebase.
        qkv[:, :, :, : self.dim] += new_q  # add to q slice
        qkv[:, :, :, -self.dim :] += new_v  # add to v slice
        # k slice is left unchanged

        qkv = self.lora_dropout(qkv)

        return qkv


# _LoRA_linear: Apply LoRA on a generic linear projection (e.g., MLP fc).
class _LoRA_linear(nn.Module):
    """
    Attributes:
        fc (nn.Module): The original linear layer.
        linear_in_er (nn.Module): LoRA A matrix (input -> rank).
        linear_er_hidden (nn.Module): LoRA B matrix (rank -> output).
        lora_dropout: Dropout (or no-op) after adding LoRA increment.
    """

    def __init__(
        self,
        fc: nn.Module,
        linear_in_er: nn.Module,
        linear_er_hidden: nn.Module,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.fc = fc
        self.linear_in_er = linear_in_er
        self.linear_er_hidden = linear_er_hidden

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): (B, N, C) input.

        Returns:
            Tensor: fc(x) augmented with LoRA increment.
        """
        fc = self.fc(x)
        new_in_hidden = self.linear_er_hidden(self.linear_in_er(x))
        fc += new_in_hidden
        fc = self.lora_dropout(fc)

        return fc


# _LoRA_qkv_proj: LoRA adapter for a projection linear layer (e.g., output proj after attention).
class _LoRA_qkv_proj(nn.Module):
    """
    Attributes:
        proj (nn.Module): The original projection layer.
        w_a (nn.Module): LoRA A matrix.
        w_b (nn.Module): LoRA B matrix.
        lora_dropout: Dropout (or no-op) after adding LoRA increment.
    """
    def __init__(self, proj: nn.Module, w_a: nn.Module, w_b: nn.Module, lora_dropout: float = 0.0):
        super().__init__()
        self.proj = proj
        self.w_a = w_a
        self.w_b = w_b
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = lambda x: x

    def forward(self, x):
        """Forward pass: proj(x) + LoRA(x)."""
        x = self.proj(x) + self.w_b(self.w_a(x))
        x = self.lora_dropout(x)
        return x


# LoRA: Inject LoRA modules into a SAM-like ViT image encoder to reduce trainable parameters
# while preserving performance.
class LoRA(nn.Module):
    r"""Apply low-rank adaptation to a SAM model's image encoder.

    Args:
        sam_model (CellViT_Instanseg): A ViT-like model (see your base encoder).
        config (dict): Configuration controlling which parts to freeze, LoRA ranks, dropout, etc.
            Required keys:
                - "freeze_image_encoder" (bool)
                - "image_encoder_lora_rank" (int)
                - "lora_dropout" (float)
        lora_layer (list or None): Indices of encoder blocks to apply LoRA.
            If None, apply LoRA to all blocks in `sam_model.encoder.blocks`.
        zero_initial (bool): If True, initialize LoRA A to zeros (B always zeros).
                             Otherwise, use Kaiming uniform for A and zeros for B.

    Example:
        >>> lora_model = LoRA(sam_model, config={"freeze_image_encoder": True,
        ...                                      "image_encoder_lora_rank": 4,
        ...                                      "lora_dropout": 0.0})
        >>> out = lora_model(img)
    """

    """
    Internal notes:
        - sam_model: pretrained SAM-like encoder/decoder model (ViT-style image encoder).
        - If `freeze_image_encoder` is True, all encoder params are frozen except for explicitly
          allowed adapters (names checked below).
        - LoRA is injected into the attention qkv linear of selected blocks.
    """

    def __init__(self, sam_model: CellViT_Instanseg, config, lora_layer=None, zero_initial=False):
        super(LoRA, self).__init__()

        # If lora_layer is not provided, apply LoRA to all transformer blocks in the image encoder.
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(range(len(sam_model.encoder.blocks)))

        self.config = config

        # Storage for LoRA A/B matrices (for saving/loading)
        self.w_As = []  # list of nn.Linear (A)
        self.w_Bs = []  # list of nn.Linear (B)

        # Optionally freeze encoder parameters except for a whitelist of adapters/heads.
        if self.config["freeze_image_encoder"]:
            for name, param in sam_model.encoder.named_parameters():
                if not (
                    any(name.startswith(k) for k in
                        ["fine", "cnn_embed", "cat_key1", "cnn_out1", "cnn_out2", "cnn_out3",
                         "CNN_ViTs", "ViT_CNNs", "ffns"]) or
                    "refine_Adapter" in name or
                    "Space_Adapter" in name
                ):
                    param.requires_grad = False

        # Inject LoRA into the image encoder (attention qkv) with given rank `er`.
        er = self.config["image_encoder_lora_rank"]

        if er > 0:
            for t_layer_i, blk in enumerate(sam_model.encoder.blocks):

                # Apply LoRA only to selected layers
                if t_layer_i not in self.lora_layer:
                    continue

                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features

                # LoRA A/B for Q
                w_a_linear_q = nn.Linear(self.dim, er, bias=False)
                w_b_linear_q = nn.Linear(er, self.dim, bias=False)

                # LoRA A/B for V
                w_a_linear_v = nn.Linear(self.dim, er, bias=False)
                w_b_linear_v = nn.Linear(er, self.dim, bias=False)

                # Keep references for save/load
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)

                # Replace attention qkv by LoRA-augmented module
                blk.attn.qkv = _LoRA_qkv(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                    lora_dropout=self.config["lora_dropout"],
                )

        # Initialize LoRA parameters (A, B) per config
        self.reset_parameters(zero_initial=zero_initial)

        # Keep a handle to the wrapped SAM-like model
        self.sam = sam_model

    def save_lora_parameters(self, filename) -> None:
        r"""Save LoRA A/B weights to a .pt/.pth file.

        Note:
            The message about "safetensors only" in the original comment is not enforced here.
            This function uses torch.save for simplicity.
        """
        filename = str(filename)
        assert filename.endswith(".pt") or filename.endswith(".pth")

        num_layer = len(self.w_As)  # number of LoRA linear modules
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}

        merged_dict = {**a_tensors, **b_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename) -> None:
        r"""Load LoRA A/B weights from a .pt/.pth file and assign to the current model."""
        filename = str(filename)
        assert filename.endswith(".pt") or filename.endswith(".pth")

        state_dict = torch.load(filename, weights_only=True)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        # Reload the full wrapped model's state dict (no external side effects here)
        sam_dict = self.sam.state_dict()
        self.sam.load_state_dict(sam_dict)

    @staticmethod
    def kaiming_uniform_5(tensor) -> None:
        """Initialize a tensor with Kaiming uniform (He) distribution."""
        nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

    def reset_parameters(self, zero_initial=False) -> None:
        """Initialize LoRA parameters.

        If zero_initial is True:
            - A is initialized to all zeros (so LoRA starts as a no-op).
            - B is always zeros.
        Otherwise:
            - A uses Kaiming uniform, B is zeros.
        """

        if zero_initial:
            initial_func = nn.init.zeros_
        else:
            initial_func = self.kaiming_uniform_5

        if self.config["image_encoder_lora_rank"] > 0:
            for w_A in self.w_As:
                initial_func(w_A.weight)
            for w_B in self.w_Bs:
                nn.init.zeros_(w_B.weight)

    def forward(self, x):

        """Forward to the wrapped SAM-like model."""

        return self.sam(x)



if __name__ == "__main__":

    print("Start")
    # sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    # lora_sam = LoRA_Sam(sam, 4)
    # lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))

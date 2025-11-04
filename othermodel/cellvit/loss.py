import torch
import torch.nn.functional as F
from typing import List, Tuple
from torch import nn
from torch.nn.modules.loss import _Loss

class XentropyLoss(_Loss):
    """Cross entropy loss"""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(size_average=None, reduce=None, reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NCHW shape of array, must be torch.float32 dtype

        Args:
            input (torch.Tensor): Ground truth array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Prediction array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Cross entropy loss, with shape () [scalar], grad_fn = MeanBackward0
        """
        # reshape
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)

        epsilon = 10e-8
        # scale preds so that the class probs of each sample sum to 1
        pred = input / torch.sum(input, -1, keepdim=True)
        # manual computation of crossentropy
        pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
        loss = -torch.sum((target * torch.log(pred)), -1, keepdim=True)
        loss = loss.mean() if self.reduction == "mean" else loss.sum()

        return loss


class DiceLoss(_Loss):
    """Dice loss

    Args:
        smooth (float, optional): Smoothing value. Defaults to 1e-3.
    """

    def __init__(self, smooth: float = 1e-3) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")
        self.smooth = smooth

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Assumes NCHW shape of array, must be torch.float32 dtype

        `pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC.

        Args:
            input (torch.Tensor): Prediction array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes
            target (torch.Tensor): Ground truth array with shape (N, C, H, W) with N being the batch-size, H the height, W the width and C the number of classes

        Returns:
            torch.Tensor: Dice loss, with shape () [scalar], grad_fn=SumBackward0
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        inse = torch.sum(input * target, (0, 1, 2))
        l = torch.sum(input, (0, 1, 2))
        r = torch.sum(target, (0, 1, 2))
        loss = 1.0 - (2.0 * inse + self.smooth) / (l + r + self.smooth)
        loss = torch.sum(loss)

        return loss


class MSELossMaps(_Loss):
    """Calculate mean squared error loss for combined horizontal and vertical maps of segmentation tasks."""

    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Loss calculation

        Args:
            input (torch.Tensor): Prediction of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal
            target (torch.Tensor): Ground truth of combined horizontal and vertical maps
                with shape (N, 2, H, W), channel 0 is vertical and channel 1 is horizontal

        Returns:
            torch.Tensor: Mean squared error per pixel with shape (N, 2, H, W), grad_fn=SubBackward0

        """
        # reshape
        loss = input - target
        loss = (loss * loss).mean()
        return loss


class MSGELossMaps(_Loss):
    def __init__(self) -> None:
        super().__init__(size_average=None, reduce=None, reduction="mean")

    def get_sobel_kernel(
        self, size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sobel kernel with a given size.

        Args:
            size (int): Kernel site
            device (str): Cuda device

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Horizontal and vertical sobel kernel, each with shape (size, size)
        """
        assert size % 2 == 1, "Must be odd, get size=%d" % size

        h_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        v_range = torch.arange(
            -size // 2 + 1,
            size // 2 + 1,
            dtype=torch.float32,
            device=device,
            requires_grad=False,
        )
        h, v = torch.meshgrid(h_range, v_range, indexing="ij")
        kernel_h = h / (h * h + v * v + 1.0e-15)
        kernel_v = v / (h * h + v * v + 1.0e-15)
        return kernel_h, kernel_v

    def get_gradient_hv(self, hv: torch.Tensor, device: str) -> torch.Tensor:
        """For calculating gradient of horizontal and vertical prediction map


        Args:
            hv (torch.Tensor): horizontal and vertical map
            device (str): CUDA device

        Returns:
            torch.Tensor: Gradient with same shape as input
        """
        kernel_h, kernel_v = self.get_sobel_kernel(5, device=device)
        kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
        kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

        h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
        v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

        # can only apply in NCHW mode
        h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
        v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
        dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
        dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC
        return dhv

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        focus: torch.Tensor,
        device: str,
    ) -> torch.Tensor:
        """MSGE (Gradient of MSE) loss

        Args:
            input (torch.Tensor): Input with shape (B, C, H, W)
            target (torch.Tensor): Target with shape (B, C, H, W)
            focus (torch.Tensor): Focus, type of masking (B, C, W, W)
            device (str): CUDA device to work with.

        Returns:
            torch.Tensor: MSGE loss
        """
        input = input.permute(0, 2, 3, 1)
        target = target.permute(0, 2, 3, 1)
        focus = focus.permute(0, 2, 3, 1)
        focus = focus[..., 1]

        focus = (focus[..., None]).float()  # assume input NHW
        focus = torch.cat([focus, focus], axis=-1).to(device)
        true_grad = self.get_gradient_hv(target, device)
        pred_grad = self.get_gradient_hv(input, device)
        loss = pred_grad - true_grad
        loss = focus * (loss * loss)
        # artificial reduce_mean with focused region
        loss = loss.sum() / (focus.sum() + 1.0e-8)
        return loss



from typing import Union

def retrieve_loss_fn(name: str) -> Union[torch.nn.Module, callable]:

    name = name.lower()

    if name == "xentropy_loss":
        return XentropyLoss()
    elif name == "dice_loss":
        return DiceLoss()
    elif name == "mse_loss_maps":
        return MSELossMaps()
    elif name == "msge_loss_maps":
        return MSGELossMaps()
    else:
        raise ValueError(f"Unknown loss function name: {name}")



def compute_total_loss(predictions, gt, loss_fn_dict, device):
    total_loss = 0.0

    for loss_name, loss_info in loss_fn_dict["nuclei_binary_map"].items():
        loss_fn = loss_info["loss_fn"]
        weight = loss_info["weight"]
        loss_val = loss_fn(predictions["nuclei_binary_map"], gt["nuclei_binary_map"])
        total_loss += weight * loss_val

    for loss_name, loss_info in loss_fn_dict["hv_map"].items():
        loss_fn = loss_info["loss_fn"]
        weight = loss_info["weight"]
        if loss_name == "msge":
            focus = gt["nuclei_binary_map"]
            loss_val = loss_fn(predictions["hv_map"], gt["hv_map"], focus, device)
        else:
            loss_val = loss_fn(predictions["hv_map"], gt["hv_map"])
        total_loss += weight * loss_val

    return total_loss


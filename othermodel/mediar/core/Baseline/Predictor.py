import torch
import os, sys
from skimage import morphology, measure
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from othermodel.mediar.core.BasePredictor import BasePredictor

__all__ = ["Predictor"]


class Predictor(BasePredictor):
    def __init__(
        self,
        model,
        device,
        input_path,
        output_path,
        make_submission=False,
        exp_name=None,
        algo_params=None,
    ):
        super(Predictor, self).__init__(
            model,
            device,
            input_path,
            output_path,
            make_submission,
            exp_name,
            algo_params,
        )

    def _inference(self, img_data):
        # 调用 sliding_window_inference() 函数，对 img_data 进行推理，预测出图像的掩码（pred_mask）。
        # sliding_window_inference() 是用于对大图像进行分块推理的函数，尤其适用于内存受限或图像尺寸较大的情况。
        # 它通过滑动窗口的方式对输入图像逐步进行局部推理，然后将结果拼接在一起得到最终预测掩码。
        pred_mask = sliding_window_inference(
            img_data,
            512, # 这个数字 512 表示滑动窗口的尺寸，也就是说在推理过程中，每次处理的图像区域为 512x512 像素。
            4, # 滑动窗口的 batch 大小，表示每次推理时，模型可以同时处理 4 个 512x512 的子图像。
            self.model,
            padding_mode="constant", # 指定在窗口边界进行填充的方式。当滑动窗口接近图像边界时，如果窗口超出了图像尺寸，就需要对图像边界进行填充。
            mode="gaussian", # "gaussian" 表示使用高斯加权平均的方式对重叠区域进行融合，较为常用，因为它可以平滑过渡，减少边缘效应。
            overlap=0.6,  # overlap 是滑动窗口之间的重叠度，表示每个窗口之间有 60% 的重叠区域。
        )

        return pred_mask

    # 对模型输出的预测掩码进行后处理（Post-Processing），包括从逻辑值转换为概率图，应用形态学操作，最终生成处理后的掩码
    def _post_process(self, pred_mask):
        # Get probability map from the predicted logits

        # 将 numpy 格式的 pred_mask 转换为 PyTorch 的张量（tensor）
        pred_mask = torch.from_numpy(pred_mask)

        # 对 pred_mask 进行 softmax 操作，将其从逻辑值（logits）转换为概率值
        pred_mask = torch.softmax(pred_mask, dim=0)

        # 从 softmax 输出的结果中提取第二个通道（通常表示前景类）。
        # [1] 表示选择 softmax 输出中第 1 个类别的概率图，例如前景对象的概率。
        # .cpu() 将张量从 GPU 移到 CPU（如果在 GPU 上执行的话），然后 .numpy() 将 PyTorch 张量转换回 numpy 数组格式，以便进行后续的 numpy 操作。
        pred_mask = pred_mask[1].cpu().numpy()

        # Apply morphological post-processing

        # 将概率图中的值进行阈值处理，生成二值化的掩码。
        # > 0.5 表示将大于 0.5 的概率视为前景，将小于等于 0.5 的概率视为背景。
        # 这一步生成了一个二值化的掩码图，前景部分为 True，背景部分为 False。
        pred_mask = pred_mask > 0.5

        # 使用形态学操作去除掩码中的小孔洞，保持连通性为 1
        pred_mask = morphology.remove_small_holes(pred_mask, connectivity=1)

        # 使用形态学操作去除掩码中小的连通区域（小物体），以减少噪声。
        # remove_small_objects 函数可以删除小的独立物体，16 表示小于 16 个像素的区域将被移除
        pred_mask = morphology.remove_small_objects(pred_mask, 16)

        # 对二值化的掩码进行连通区域标记。
        pred_mask = measure.label(pred_mask)

        return pred_mask

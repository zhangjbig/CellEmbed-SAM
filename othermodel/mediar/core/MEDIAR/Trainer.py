import torch
import torch.nn as nn
import numpy as np
import os, sys
from tqdm import tqdm
from monai.inferers import sliding_window_inference

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))

from othermodel.mediar.core.BaseTrainer import BaseTrainer
from othermodel.mediar.core.MEDIAR.utils import *

__all__ = ["Trainer"]


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        dataloaders,
        optimizer,
        scheduler=None,
        criterion=None,
        num_epochs=100,
        device="cuda:0",
        no_valid=False,
        valid_frequency=1,
        amp=False,
        algo_params=None,
    ):
        super(Trainer, self).__init__(
            model,
            dataloaders,
            optimizer,
            scheduler,
            criterion,
            num_epochs,
            device,
            no_valid,
            valid_frequency,
            amp,
            algo_params,
        )

        self.mse_loss = nn.MSELoss(reduction="mean")
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        # self.with_public = True

    def mediar_criterion(self, outputs, labels_onehot_flows):
        """loss function between true labels and prediction outputs"""


        # Cell Recognition Loss
        """
        计算细胞识别的损失。
        outputs[:, -1]：提取模型输出中的细胞概率（最后一个通道）。
        torch.from_numpy(labels_onehot_flows[:, 1] > 0.5)：将标签转换为张量，标记细胞的存在（大于 0.5 的值视为细胞）。
        .to(self.device).float()：将标签张量移动到指定的设备，并转换为浮点类型。
        self.bce_loss：使用二元交叉熵损失函数计算损失。
        """
        cellprob_loss = self.bce_loss(
            outputs[:, -1],
            torch.from_numpy(labels_onehot_flows[:, 1] > 0.5).to(self.device).float(),
        )

        # Cell Distinction Loss
        """
        计算细胞区分的损失。
        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(self.device)：将标签中的梯度流部分（第 2 个通道及之后的部分）转换为张量并移动到设备。
        outputs[:, :2]：提取模型输出中的前两个通道（表示梯度流）。
        5.0 * gradient_flows：将梯度流标签放大 5 倍，以增强对模型的训练信号。
        self.mse_loss：使用均方误差损失函数计算输出与标签之间的损失。
        gradflow_loss = 0.5 * ...：将计算得到的损失乘以 0.5。
        """
        gradient_flows = torch.from_numpy(labels_onehot_flows[:, 2:]).to(self.device)
        gradflow_loss = 0.5 * self.mse_loss(outputs[:, :2], 5.0 * gradient_flows)

        loss = cellprob_loss + gradflow_loss

        return loss

    # 定义一个名为 _epoch_phase 的方法，接收一个参数 phase，表示当前阶段（如训练或验证）。
    def _epoch_phase(self, phase):
        # 初始化一个空字典 phase_results 用于存储当前阶段的结果。
        phase_results = {}

        # Set model mode
        """
        据 phase 的值设置模型的模式：
        如果是训练阶段 (train)，则调用 self.model.train() 进入训练模式。
        否则，调用 self.model.eval() 进入评估模式。
        """
        self.model.train() if phase == "train" else self.model.eval()

        # Epoch process 使用 tqdm 包装数据加载器，遍历当前阶段的数据集。batch_data 包含每个批次的数据。
        for batch_data in tqdm(self.dataloaders[phase]):
            # 从 batch_data 中提取图像和标签。
            images, labels = batch_data["img"], batch_data["label"]

            # 如果 self.with_public 为真，说明需要从未标记的数据加载器加载公共数据。
            if self.with_public:
                # Load batches sequentially from the unlabeled dataloader

                # 尝试从公共数据迭代器中获取下一个批次的数据。
                try:
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]
                # 如果公共数据迭代器已结束，重新创建迭代器并加载下一个批次的数据。
                except:
                    # Assign memory loader if the cycle ends
                    self.public_iterator = iter(self.public_loader)
                    batch_data = next(self.public_iterator)
                    images_pub, labels_pub = batch_data["img"], batch_data["label"]

                # Concat memory data to the batch 将加载的公共图像和标签与当前批次的图像和标签进行拼接。
                images = torch.cat([images, images_pub], dim=0)
                labels = torch.cat([labels, labels_pub], dim=0)

            # 将图像和标签数据移动到指定的设备（如 GPU）
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 在每个批次开始前清零优化器的梯度，以防止累积。
            self.optimizer.zero_grad()

            # Forward pass 前向传播

            # 启用混合精度训练（如果 self.amp 为真），并在训练阶段启用梯度计算。
            """
            torch.cuda.amp 是 PyTorch 中用于混合精度训练的模块。
            autocast 是一个上下文管理器，用于自动将计算放在混合精度模式下进行。即在这个上下文中，PyTorch 会根据需要自动执行 FP16（半精度浮点数）计算。
            enabled=self.amp 表示是否启用混合精度训练，self.amp 是一个布尔值，控制是否启用混合精度。
            """
            with torch.cuda.amp.autocast(enabled=self.amp):

                """
                torch.set_grad_enabled() 是一个上下文管理器，用于控制是否开启梯度计算。
                phase == "train" 是条件判断，表示在训练阶段（phase 为 "train"）时才开启梯度计算。
                当 phase == "train" 时，梯度计算被开启；否则，在验证或测试阶段，梯度计算被关闭。
                """
                with torch.set_grad_enabled(phase == "train"):
                    # Output shape is B x [grad y, grad x, cellprob] x H x W
                    # 调用 _inference 方法进行前向传播，得到模型的输出。
                    outputs = self._inference(images, phase)

                    # Map label masks to graidnet and onehot 将标签转换为 梯度流和one-hot 编码格式，适应后续损失计算。
                    labels_onehot_flows = labels_to_flows(
                        labels, use_gpu=True, device=self.device
                    )

                    # Calculate loss 使用指定的损失函数 self.mediar_criterion 计算损失，并将损失添加到损失指标列表中。
                    loss = self.mediar_criterion(outputs, labels_onehot_flows)
                    self.loss_metric.append(loss)

                    # Calculate valid statistics 如果当前阶段不是训练阶段，对输出和标签进行后处理，并计算 F1 分数，添加到 F1 指标列表中。
                    if phase != "train":
                        outputs, labels = self._post_process(outputs, labels)
                        f1_score = self._get_f1_metric(outputs, labels)
                        self.f1_metric.append(f1_score)

                # Backward pass 如果当前阶段是训练阶段，进行反向传播。
                if phase == "train":
                    # For the mixed precision training
                    """
                    如果启用混合精度：
                    使用 scaler 进行损失的缩放和反向传播。
                    更新优化器。
                    """
                    if self.amp:
                        self.scaler.scale(loss).backward() # 使用 scaler.scale(loss) 将损失进行缩放，以防止数值下溢。调用 backward() 方法进行反向传播。
                        self.scaler.unscale_(self.optimizer) # 调用 unscale_() 将优化器的梯度解缩放。
                        self.scaler.step(self.optimizer) # 使用 step() 更新优化器参数。
                        self.scaler.update() # 调用 update() 更新 scaler 状态
                    # 如果没有启用混合精度，直接进行常规的反向传播和优化器更新。
                    else:
                        loss.backward()
                        self.optimizer.step()

        # Update metrics 更新当前阶段的损失指标到 phase_results 字典中。
        phase_results = self._update_results(
            phase_results, self.loss_metric, "dice_loss", phase
        )
        if phase != "train": # 如果当前阶段不是训练阶段，将 F1 指标添加到 phase_results 中。
            phase_results = self._update_results(
                phase_results, self.f1_metric, "f1_score", phase
            )

        return phase_results

    def _inference(self, images, phase="train"):
        """
        inference methods for different phase
        使用 sliding_window_inference 函数进行推理，以处理较大的图像。
        images：输入图像。
        roi_size=512：设置感兴趣区域的大小为 512x512。
        sw_batch_size=4：设置滑动窗口推理时的批次大小为 4。
        predictor=self.model：使用当前模型进行预测。
        padding_mode="constant"：使用常数填充模式。
        mode="gaussian"：指定推理模式为高斯模式。
        overlap=0.5：设置滑动窗口的重叠比例为 50%。
        """
        if phase != "train":
            outputs = sliding_window_inference(
                images,
                roi_size=512,
                sw_batch_size=4,
                predictor=self.model,
                padding_mode="constant",
                mode="gaussian",
                overlap=0.5,
            )
        else:
            outputs = self.model(images)

        return outputs

    def _post_process(self, outputs, labels=None):
        """Predict cell instances using the gradient tracking"""

        # 将输入 outputs 的第一个维度去掉（即 squeeze(0)，
        # 假设 outputs 的形状为 (1, C, H, W)），将其转换为 (C, H, W)，
        # 然后将其移动到 CPU 上，并转换为 NumPy 数组，方便后续的操作。
        outputs = outputs.squeeze(0).cpu().numpy()

        # 对 outputs 进行切片，前两个通道（outputs[:2]）赋值给 gradflows，最后一个通道（outputs[-1]）经过 sigmoid 激活后赋值给 cellprob。
        gradflows, cellprob = outputs[:2], self._sigmoid(outputs[-1])

        # 调用 compute_masks 函数，利用 gradflows 和 cellprob 生成细胞掩膜。这里可能会使用 GPU（通过 use_gpu=True）和指定的 device。
        outputs = compute_masks(gradflows, cellprob, use_gpu=True, device=self.device)

        # 从 compute_masks 的输出中提取第一个维度。假设 compute_masks 的输出形状为 (1, C, H, W)，则 outputs[0] 将移除第一个维度，得到 (C, H, W)。
        outputs = outputs[0]  # (1, C, H, W) -> (C, H, W)

        if labels is not None:
            labels = labels.squeeze(0).squeeze(0).cpu().numpy()

        return outputs, labels

    def _sigmoid(self, z):
        """Sigmoid function for numpy arrays"""
        return 1 / (1 + np.exp(-z))

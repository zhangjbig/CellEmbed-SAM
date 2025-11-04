import torch
import numpy as np
import time, os
import tifffile as tif

from datetime import datetime
from zipfile import ZipFile
from pytz import timezone

from othermodel.mediar.data_utils.transforms import get_pred_transforms


class BasePredictor:
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
        self.model = model
        self.device = device
        self.input_path = input_path
        self.output_path = output_path
        self.make_submission = make_submission
        self.exp_name = exp_name

        # Assign algoritm-specific arguments 将字典中的每个键值对直接设置为对象的属性和值
        if algo_params:
            self.__dict__.update((k, v) for k, v in algo_params.items())

        # Prepare inference environments
        self._setups()

    @torch.no_grad()
    def conduct_prediction(self):

        # 将模型移动到指定的设备（通常是CPU或GPU）。如果self.device是'cuda'，模型会被移到GPU上，否则在CPU上运行。
        self.model.to(self.device)

        # 将模型设置为评估模式（evaluation mode）。
        # 在评估模式下，某些操作（如Dropout和BatchNorm）将表现为推断时的行为，而不是训练时的行为，确保模型在测试时表现稳定。
        self.model.eval()

        total_time = 0 # 用于累加所有图像的推断时间。
        total_times = [] # 记录每个图像的推断时间的列表。

        # 遍历self.img_names中的每个图像名称，进行逐个图像的推断。
        for img_name in self.img_names:

            # 通过调用_get_img_data方法，根据图像名称加载该图像的数据
            img_data = self._get_img_data(img_name)
            # 将加载的图像数据移动到指定的设备上（通常是GPU），以便在设备上进行推断。
            img_data = img_data.to(self.device)

            # 记录当前时间，作为开始时间，用于后续计算每张图像的推断耗时
            start = time.time()

            # 调用_inference方法进行推断，将图像数据输入模型，得到预测的掩码（pred_mask）。
            pred_mask = self._inference(img_data)

            # 对预测的掩码进行后处理：
            # squeeze(0)：去掉预测掩码的第一个维度（批次维度）。
            # cpu().numpy()：将掩码数据从GPU移到CPU，并转换为NumPy数组格式。
            # self._post_process：对掩码进行进一步的后处理操作（如二值化、过滤等）。
            pred_mask = self._post_process(pred_mask.squeeze(0).cpu().numpy())

            # 调用write_pred_mask方法，将处理后的预测掩码保存到指定路径（self.output_path）下。
            # self.make_submission决定是否需要保存掩码用于提交。
            self.write_pred_mask(
                pred_mask, self.output_path, img_name, self.make_submission
            )

            # 记录当前时间，作为推断结束时间。
            end = time.time()

            # 计算单张图像的推断耗时，time_cost表示该图像的推断时间。
            time_cost = end - start
            # 将该图像的推断时间添加到total_times列表中。
            total_times.append(time_cost)
            # 累加每张图像的推断时间，得到所有图像的总推断时间。
            total_time += time_cost

            # 打印当前图像的名称、尺寸和推断时间，格式化输出保留两位小数。
            print(
                f"Prediction finished: {img_name}; img size = {img_data.shape}; costing: {time_cost:.2f}s"
            )

        # 当所有图像都完成推断后，打印所有图像的总推断时间。
        print(f"\n Total Time Cost: {total_time:.2f}s")

        # 检查是否需要生成提交文件。如果self.make_submission为True，进入提交文件生成步骤。
        if self.make_submission:
            # 创建一个用于保存提交结果的zip文件名，self.exp_name是实验名称。
            fname = "%s.zip" % self.exp_name

            # 创建一个名为./submissions的目录（如果不存在的话）。
            os.makedirs("./submissions", exist_ok=True)
            submission_path = os.path.join("./submissions", fname)

            # 打开一个Zip文件对象（zipObj2），准备写入文件到submission_path路径。
            with ZipFile(submission_path, "w") as zipObj2:
                # 获取输出路径（self.output_path）下所有预测掩码文件的名称，并进行排序。
                pred_names = sorted(os.listdir(self.output_path))
                for pred_name in pred_names:
                    pred_path = os.path.join(self.output_path, pred_name)
                    zipObj2.write(pred_path)

            print("\n>>>>> Submission file is saved at: %s\n" % submission_path)

        return time_cost

    def write_pred_mask(self, pred_mask, output_dir, image_name, submission=False):

        # All images should contain at least 5 cells
        if submission:
            if not (np.max(pred_mask) > 5):
                print("[!Caution] Only %d Cells Detected!!!\n" % np.max(pred_mask))

        file_name = image_name.split(".")[0]
        file_name = file_name + "_label.tiff"
        file_path = os.path.join(output_dir, file_name)

        tif.imwrite(file_path, pred_mask, compression="zlib")

    def _setups(self):
        # 通常用于获取预测任务所需要的图像变换（transformation）。这些变换可以是预处理操作，比如缩放、归一化等，适用于模型推理过程。
        self.pred_transforms = get_pred_transforms()

        # 通过 os.makedirs() 函数，创建一个输出目录，路径为 self.output_path。
        # exist_ok=True 表示如果该目录已经存在，不会抛出错误，代码会继续执行。
        os.makedirs(self.output_path, exist_ok=True)

        # 获取当前时间，并将其设置为 Asia/Seoul（首尔时间）的时区。
        # datetime.now() 获取当前时间，timezone("Asia/Seoul") 用于指定时区。
        now = datetime.now(timezone("Asia/Seoul"))
        # 将当前时间格式化为一个字符串，格式为：月日_小时分钟。
        dt_string = now.strftime("%m%d_%H%M")

        self.exp_name = (
            self.exp_name + dt_string if self.exp_name is not None else dt_string
        )

        # 使用 os.listdir() 获取 self.input_path 目录下所有文件的文件名，并将它们排序后赋给 self.img_names。
        # sorted() 是 Python 的排序函数，这里是对输入文件名进行字母或数字的排序，确保按照某种顺序处理输入图像文件。
        self.img_names = sorted(os.listdir(self.input_path))

    def _get_img_data(self, img_name):
        img_path = os.path.join(self.input_path, img_name)
        img_data = self.pred_transforms(img_path)
        img_data = img_data.unsqueeze(0)

        return img_data

    def _inference(self, img_data):
        raise NotImplementedError

    def _post_process(self, pred_mask):
        raise NotImplementedError

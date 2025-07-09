import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path

import torch
from torch.backends import cuda, cudnn

cuda.matmul.allow_tf32 = True
cudnn.allow_tf32 = True
torch.multiprocessing.set_sharing_strategy("file_system")

import hydra
import hdf5plugin
from omegaconf import DictConfig, OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
# ----------------- 新增导入项 -----------------
import cv2
import numpy as np
from lightning.pytorch.callbacks import ModelSummary, Callback
from utils.evaluation.prophesee.visualize.vis_utils import (
    LABELMAP_GEN1,
    LABELMAP_GEN4_SHORT,
    draw_bboxes,
)
from modules.detection import ObjDetOutput # 假设 ObjDetOutput 在此模块
# ----------------------------------------------------

from config.modifier import dynamically_modify_train_config
from modules.utils.fetch import fetch_data_module, fetch_model_module
from modules.detection import Module
from einops import rearrange, reduce

# ----------------- 修复后的 Callback 类 -----------------
class VisualizationCallback(Callback):
    """
    在每个测试步骤后生成并保存可视化结果的回调。
    V3: 修复了“Boolean value of Tensor is ambiguous”的错误。
    """
    def __init__(self, save_dir="./visualization_output"):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"可视化结果将保存至: {self.save_dir.absolute()}")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """
        在每个测试批次结束后被调用。
        """
        # if ObjDetOutput.EV_REPR not in outputs:
        #     print(f"警告：在批次 {batch_idx} 的输出中未找到 'EV_REPR'。跳过可视化。")
        #     return
        
        # ev_tensors = outputs[ObjDetOutput.EV_REPR]
        
        # # ------------------- BUG 修复点 -------------------
        # # 错误原因: `if not ev_tensors:` 试图将整个张量作为布尔值，这是不允许的。
        # # 正确方法: 检查张量的第一个维度（样本数量）是否为0。
        # if ev_tensors.shape[0] == 0:
        #     print(f"警告: 在批次 {batch_idx} 的输出中 'EV_REPR' 不包含任何样本。跳过可视化。")
        #     return
        # ----------------------------------------------------

        # # 我们将可视化序列中的最后一个样本
        # sample_idx = ev_tensors.shape[0] - 1
        # ev_tensor_sample = ev_tensors[sample_idx]
        def ev_repr_to_img(x: np.ndarray):
            # print(f"ev_repr_to_img input shape: {x.shape}")
            if x.ndim == 2:
                # 形状是 (H, W)，这里暂时假设通道是1
                ht, wd = x.shape
                ch = 1
                # 给channel维度补上，方便后续逻辑
                x = x[np.newaxis, :, :]  # (1, H, W)
            elif x.ndim == 3:
                ch, ht, wd = x.shape
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

            # 如果ch是1，说明没有pos/neg通道，直接返回灰度图或二值图
            if ch == 1:
                img = (x[0] * 255).astype(np.uint8)  # 简单放大到0~255
                img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                return img_color

            # 你原先预期是ch > 1 并且是偶数
            assert ch > 1 and ch % 2 == 0

            ev_repr_reshaped = rearrange(x, "(posneg C) H W -> posneg C H W", posneg=2)
            img_neg = np.asarray(
                reduce(ev_repr_reshaped[0], "C H W -> H W", "sum"), dtype="int32"
            )
            img_pos = np.asarray(
                reduce(ev_repr_reshaped[1], "C H W -> H W", "sum"), dtype="int32"
            )
            img_diff = img_pos - img_neg
            img = 127 * np.ones((ht, wd, 3), dtype=np.uint8)
            img[img_diff > 0] = 255
            img[img_diff < 0] = 0
            return img       
        # ev_img = ev_repr_to_img(ev_tensor_sample.cpu().numpy())

        # # 可视化模型的预测
        # prediction_img = ev_img.copy()
        # predictions_proph = outputs[ObjDetOutput.PRED_PROPH][sample_idx]
        # draw_bboxes(prediction_img, predictions_proph, labelmap=pl_module.label_map)

        # # 可视化真实标签 (Ground Truth)
        # label_img = ev_img.copy()
        # labels_proph = outputs[ObjDetOutput.LABELS_PROPH][sample_idx]
        # draw_bboxes(label_img, labels_proph, labelmap=pl_module.label_map)
        
        if outputs[ObjDetOutput.SKIP_VIZ]:
            return
        ev_tensor = outputs[ObjDetOutput.EV_REPR]
        assert isinstance(ev_tensor, torch.Tensor)

        ev_img = ev_repr_to_img(ev_tensor.cpu().numpy())

        predictions_proph = outputs[ObjDetOutput.PRED_PROPH]
        prediction_img = ev_img.copy()
        draw_bboxes(prediction_img, predictions_proph, labelmap=LABELMAP_GEN4_SHORT)

        labels_proph = outputs[ObjDetOutput.LABELS_PROPH]
        label_img = ev_img.copy()
        draw_bboxes(label_img, labels_proph, labelmap=LABELMAP_GEN4_SHORT)
        # 将预测图和标签图垂直合并
        final_img = np.vstack([prediction_img, label_img])
        
        # 保存最终的可视化图像
        if final_img.shape[2] == 3:
            final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
        
        save_path = self.save_dir / f"result_comparison_batch_{batch_idx:05d}.png"
        cv2.imwrite(str(save_path), final_img)


@hydra.main(config_path="config", config_name="val", version_base="1.2")
def main(config: DictConfig):
    dynamically_modify_train_config(config)
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)

    print("------ Configuration ------")
    print(OmegaConf.to_yaml(config))
    print("---------------------------")

    gpus = config.hardware.gpus
    assert isinstance(gpus, int), "no more than 1 GPU supported"
    gpus = [gpus]

    data_module = fetch_data_module(config=config)

    logger = CSVLogger(save_dir="./validation_logs")
    ckpt_path = Path(config.checkpoint)

    module = fetch_model_module(config=config)
    module = Module.load_from_checkpoint(str(ckpt_path), **{"full_config": config})

    callbacks = [ModelSummary(max_depth=2)]
    if config.use_test_set:
        callbacks.append(VisualizationCallback())

    trainer = pl.Trainer(
        accelerator="gpu",
        callbacks=callbacks,
        default_root_dir=None,
        devices=gpus,
        logger=logger,
        log_every_n_steps=100,
        precision=config.training.precision,
    )
    with torch.inference_mode():
        if config.use_test_set:
            trainer.test(model=module, datamodule=data_module, ckpt_path=str(ckpt_path))
        else:
            trainer.validate(
                model=module, datamodule=data_module, ckpt_path=str(ckpt_path)
            )

if __name__ == "__main__":
    main()

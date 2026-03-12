import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tools.cfg import py2cfg
import os
import torch
import numpy as np
import argparse
from pathlib import Path
from pytorch_lightning.loggers import CSVLogger
import random
from open_clip import tokenizer
import torchmetrics
from torch.utils.data import DataLoader
from geoseg.datasets.msesad_dataset import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, help="Path to the config.", required=True)
    return parser.parse_args()


class CLIP_Train(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = config.net
        self.loss = torch.nn.CrossEntropyLoss()   #交叉熵损失函数
        self.val_loss = torchmetrics.MeanMetric()
        self.val_macro_f1 = torchmetrics.F1Score(task="multiclass", num_classes=config.num_classes, average='macro')
        # self.val_wa = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(self.config.classes), average='weighted')
        # self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=config.num_classes, average='weighted')
        # self.val_aa = torchmetrics.classification.Accuracy(task="multiclass", num_classes=len(self.config.classes), average='macro')
        # self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=config.num_classes, average='macro')
        # self.val_confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=config.num_classes)
    def forward(self, img_rgb, img_swir):
        # only net is used in the prediction/inference
        pre = self.net(img_rgb, img_swir)
        return pre

    def training_step(self, batch, batch_idx):
        img_rgb, img_swir, label = batch['img_rgb'], batch['img_swir'], batch['cls']
        pre = self.net(img_rgb, img_swir)
        logits = pre['logits']
        train_loss = self.loss(logits, label)
        # self.train_aa(logits, label)
        # self.log('train_aa', self.train_aa, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.config.train_batch_size)
        self.log('train_loss', train_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.config.train_batch_size)
        return train_loss

    def validation_step(self, batch, batch_idx):
        img_rgb, img_swir, label = batch['img_rgb'], batch['img_swir'], batch['cls']
        res = self.net(img_rgb, img_swir)
        logits = res['logits']
        # self.val_wa(logits, label)
        # self.val_f1(logits, label)
        # self.val_aa(logits, label)
        # self.val_recall(logits, label)
        # self.val_confmat.update(logits, label)
        loss = self.loss(logits, label)
        self.val_loss.update(loss)
        self.val_macro_f1(logits, label)


    def on_validation_epoch_end(self):
        # 记录验证损失
        val_loss = self.val_loss.compute()
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.config.train_batch_size)
        val_macro_f1 = self.val_macro_f1.compute()
        self.log('val_macro_f1', val_macro_f1, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.config.val_batch_size)
        # self.log('val_loss', val_loss, prog_bar=True, batch_size=self.config.val_batch_size)
        # self.log('val_wa', self.val_wa.compute(), prog_bar=True, batch_size=self.config.val_batch_size)
        # self.log('val_f1', self.val_f1.compute(), batch_size=self.config.val_batch_size)
        # self.log('val_aa', self.val_aa.compute(), prog_bar=True, batch_size=self.config.val_batch_size)
        # self.log('val_recall', self.val_recall.compute(), batch_size=self.config.val_batch_size)

        # if (self.current_epoch + 1) % 5 == 0:
        #     cm = self.val_confmat.compute().cpu().numpy()
        #     np.save(f"confusion_matrix_epoch{self.current_epoch}.npy", cm)
        #     print(cm)
        # self.val_confmat.reset()

        self.val_loss.reset()
        self.val_macro_f1.reset()
        # self.val_wa.reset()
        # self.val_f1.reset()
        # self.val_aa.reset()
        # self.val_recall.reset()

    def configure_optimizers(self):
        self.net.train()
        optimizer = self.config.optimizer
        lr_scheduler = self.config.lr_scheduler

        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        train_loader = DataLoader(dataset=MSESADDataset(mode='train'),
                                  batch_size=self.config.train_batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  num_workers=8,
                                  persistent_workers=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(dataset=MSESADDataset(mode='test'),
                                batch_size=self.config.val_batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=8,
                                persistent_workers=True)
        return val_loader

    # def on_after_backward(self):
    #     """梯度监控回调"""
    #     # 遍历所有模块参数
    #     for name, param in self.net.named_parameters():
    #         # 只关注新增模块的梯度
    #         if 'adapter' in name:
    #             if param.grad is not None:
    #                 grad_norm = param.grad.norm().item()
    #                 self.log(f"grad_norm/{name}", grad_norm)
    #
    #                 # 检测异常梯度
    #                 if grad_norm == 0:
    #                     print(f"警告! {name} 梯度为零")
    #                 elif grad_norm > 1e5:
    #                     print(f"警告! {name} 梯度爆炸: {grad_norm:.2e}")


# training
def main():
    args = get_args()
    config = py2cfg(args.config_path)
    seed_everything(42)
    weights_path = "D:/airs/ESA_CLIP/model_weights_19".format(config.weights_name)
    logger_path = 'D:/airs/ESA_CLIP/lightning_logs'

    # 1. 创建模型检查点回调
    checkpoint_callback = ModelCheckpoint(save_top_k=config.save_top_k, monitor=config.monitor,
                                          save_last=config.save_last, mode=config.monitor_mode,
                                          dirpath=weights_path,
                                          filename=config.weights_name)
    # 2. 创建早停回调
    early_stopping = EarlyStopping(
        monitor='val_loss',  # 监控验证损失
        patience=5,  # 连续5个epoch无改善则停止
        mode='min',  # 最小化损失
        verbose=True,  # 打印早停信息
    )
    logger = CSVLogger(logger_path, name=config.log_name)

    model = CLIP_Train(config)
    if config.pretrained_ckpt_path:
        model = CLIP_Train.load_from_checkpoint(config.pretrained_ckpt_path, config=config)

    trainer = pl.Trainer(devices=config.gpus, max_epochs=config.max_epoch, accelerator='auto', precision="16-mixed",
                         check_val_every_n_epoch=config.check_val_every_n_epoch,
                         callbacks=[checkpoint_callback,early_stopping], strategy='auto',
                         logger=logger)
    trainer.fit(model=model)


if __name__ == "__main__":
   main()

import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax

from cifar10_models.clip_models import ClipViTB32
from src.simple_utils import load_pickle
from cifar10_models.Resnet import ResNet18
from cifar10_models.Densenet import DenseNet121
import pandas as pd
from typing import List, Optional
import numpy as np
import os
import fcntl

osj = os.path.join

all_classifiers = {
    "Resnet18": ResNet18,
    "Densenet121": DenseNet121,
    "ClipViTB32": ClipViTB32
}


class CIFAR10Module(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        num_classes: int = 10,
        learning_rate: float = 1e-2,
        weight_decay: float = 5e-4,
        max_epochs: int = 100,
        val_names: List[str] = [],
        pred_save_path: str = ".",
        work_dir: str = ".",
        seed: int = 1,
        hash: Optional[str] = None,
        eval_last_epoch_only: bool = False,
        pretrained: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.pred_accs = nn.ModuleList([Accuracy() for _ in val_names])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = pred_save_path
        self.work_dir = work_dir
        self.seed = seed
        self.hash = hash
        self.eval_last_epoch_only = eval_last_epoch_only
        self.pretrained = pretrained

        if pretrained:
            if arch == "Resnet18":
                pre_path = "Resnet18_epoch=99-step=500499.ckpt"
            else:
                pre_path = "Densenet121_epoch=57-step=290289.ckpt"
            self.model = all_classifiers[arch](num_classes=1000)
            state_dict = torch.load(osj(work_dir, "pretrained", pre_path))["state_dict"]
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict)
            for param in self.model.parameters():
                param.requires_grad = False
            num_ftrs = self.model.linear.in_features
            self.model.linear = nn.Linear(num_ftrs, num_classes)
        else:
            self.model = all_classifiers[arch](num_classes=num_classes)

        self.val_names = val_names
        self.is_clip = arch.startswith("Clip")

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        images, _ = batch
        return self.model(images)

    def process_batch(self, batch, stage="train", dataloader_idx=0):
        images, labels = batch
        logits = self.forward(images)
        probs = softmax(logits, dim=1)
        loss = self.criterion(logits, labels)
        if stage == "train":
            self.train_acc(probs, labels)
        elif stage == "val":
            self.valid_acc(probs, labels)
        elif stage == "pred":
            assert dataloader_idx > 0
            pred_acc = self.pred_accs[dataloader_idx - 1]
            pred_acc(probs, labels)
        return loss

    def batch_preds(self, batch):
        images, labels = batch
        logits = self.forward(images)
        preds = torch.argmax(logits, dim=1)
        preds_acc = preds == labels
        return preds_acc, labels

    def training_step(self, batch, batch_idx: int):
        loss = self.process_batch(batch, "train")
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
        return loss

    def is_eval_epoch(self):
        return (not self.eval_last_epoch_only) or (
            self.eval_last_epoch_only and self.current_epoch == (self.max_epochs - 1)
        )

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        if dataloader_idx == 0:
            loss = self.process_batch(batch, "val")
            self.log("val_loss", loss, add_dataloader_idx=False)
            self.log(
                "val_acc",
                self.valid_acc,
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        else:
            # NEEDS to be fixed for > 1 val_loaders
            loss = self.process_batch(batch, "pred", dataloader_idx)
            self.log("pred_loss", loss, add_dataloader_idx=True)
            self.log(
                "pred_acc",
                self.pred_accs[dataloader_idx - 1],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=True,
            )
            if self.is_eval_epoch():
                preds_acc, labels = self.batch_preds(batch)
                batch_info = pd.DataFrame(
                    {
                        "acc": preds_acc.detach().cpu().numpy(),
                        "label": labels.detach().cpu().numpy(),
                    }
                )
                return batch_info

    def process_outputs(self, outputs):
        if len(self.val_names) == 0:
            return

        outputs = outputs[1:]

        # save
        loc = osj(self.pred_save_path, "preds.csv")
        cats = load_pickle(osj(self.pred_save_path, "cats.pkl"))

        for idx, output in enumerate(outputs):
            df = pd.concat(output)
            df["index"] = np.arange(df.shape[0])
            df["epoch_step"] = str(self.current_epoch) + "," + str(self.global_step)
            df["hash"] = cats["hash"][self.hash]
            df["seed"] = cats["seed"][self.seed]
            df["name"] = cats["name"][self.val_names[idx]]
            df = df.sort_index(axis=1)

            if not os.path.isfile(loc):
                with open(loc, "w") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    df.to_csv(f, index=False)
                    fcntl.flock(f, fcntl.LOCK_UN)
            else:
                with open(loc, "a") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    df.to_csv(f, index=False, header=False)
                    fcntl.flock(f, fcntl.LOCK_UN)

    def validation_epoch_end(self, outputs):
        if self.is_eval_epoch():
            dir_path = osj(self.pred_save_path, "ds_avg")
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            fname = str((self.hash, self.current_epoch, self.global_step)) + ".csv"
            loc = osj(dir_path, fname)

            with open(loc, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(str(self.valid_acc.compute().item()) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)
            self.process_outputs(outputs)

    def configure_optimizers(self):
        if self.is_clip:
            parameters = self.model.linear.parameters()
        else:
            parameters = self.model.parameters()
        optimizer = torch.optim.SGD(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9
        )

        if self.pretrained:
            lr_scheduler = lr_sched.StepLR(optimizer, step_size=7, gamma=0.1)
        else:
            lr_scheduler = lr_sched.CosineAnnealingLR(optimizer, T_max=self.max_epochs)

        return [optimizer], [lr_scheduler]

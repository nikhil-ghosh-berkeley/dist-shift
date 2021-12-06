import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy 
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_sched
from torch.nn.functional import softmax
from src.simple_utils import dump_pickle, load_pickle, load_pretrained

from cifar10_models.clip_models import ClipViTB32
from cifar10_models.Resnet import ResNet18
from cifar10_models.Densenet import DenseNet121
from typing import List, Optional
import numpy as np
import os
from os.path import join, isfile
import pdb
import logging


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
        hash: Optional[str] = None,
        eval_last_epoch_only: bool = False,
        num_classes=10
        pretrained: bool = False,
        save_logits: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.model = all_classifiers[arch](num_classes=num_classes)
        self.train_acc = Accuracy()
        self.pred_accs = nn.ModuleList([Accuracy() for _ in val_names])

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = pred_save_path
        self.work_dir = work_dir
        self.hash = hash
        self.eval_last_epoch_only = eval_last_epoch_only
        self.pretrained = pretrained
        self.save_logits = save_logits

        if pretrained:
            self.model = load_pretrained(arch, work_dir, num_classes)
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
        elif stage == "pred":
            pred_acc = self.pred_accs[dataloader_idx]
            pred_acc(probs, labels)
        else:
            raise ValueError("Invalid stage %s" % stage)

        return loss

    def batch_preds(self, batch):
        images, labels = batch
        logits = self.forward(images)
        if self.save_logits:
            return logits.detach().cpu().numpy(), labels.detach().cpu().numpy()

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels)
        return acc.detach().cpu().numpy(), labels.detach().cpu().numpy()

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
        loss = self.process_batch(batch, "pred", dataloader_idx)
        self.log("pred_loss", loss, add_dataloader_idx=True)
        self.log(
            "pred_acc",
            self.pred_accs[dataloader_idx],
            on_step=False,
            on_epoch=True,
            add_dataloader_idx=True,
        )

        if self.is_eval_epoch() and dataloader_idx > 0:
            return self.batch_preds(batch)

    # outputs is a list of |val_names| lists of len num_batches, with first list empty
    def process_outputs(self, outputs):
        pred_type = "logits" if self.save_logits else "acc"
        outputs = outputs[1:]

        # save
        loc = join(self.pred_save_path, "raw")
        for idx, output in enumerate(outputs):
            val_name = self.val_names[idx + 1]
            fname = str((val_name, self.hash, self.current_epoch, self.global_step))
            out = np.concatenate([p[0] for p in output], axis=0)
            label = np.concatenate([p[1] for p in output], axis=0)

            if isfile(join(loc, fname)):
                dct = load_pickle(join(loc, fname))
            else:
                dct = {pred_type: [], "label": [], "index": [], "dset": []}

            dct[pred_type].append(out)
            dct["label"].append(label)
            dct["index"].append(np.arange(len(label)))
            dump_pickle(dct, join(loc, fname))

    # process all validation batch outputs
    def validation_epoch_end(self, outputs):
        loc = join(self.pred_save_path, "raw")
        if self.is_eval_epoch() and len(self.val_names) > 1:
            self.process_outputs(outputs)
            # add dataset_idx[0] statistics
            for val_name in self.val_names[1:]:
                fname = str((val_name, self.hash, self.current_epoch, self.global_step))
                dct = load_pickle(join(loc, fname))
                dct['dset'].append(self.pred_accs[0].compute().item())
                dump_pickle(dct, join(loc, fname))

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

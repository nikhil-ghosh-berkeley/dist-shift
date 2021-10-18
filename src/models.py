import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch
from torch.nn.functional import softmax
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
}

class CIFAR10Module(pl.LightningModule):
    def __init__(
        self,
        arch: str = "Resnet18",
        learning_rate: float = 1e-2,
        weight_decay: float = 5e-4,
        max_epochs: int = 100,
        val_names: List[str] = [],
        pred_save_path: str = ".",
        seed: int = 1,
        hash: Optional[str] = None,
        eval_last_epoch_only: bool = False,
    ):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy()
        self.model = all_classifiers[arch]()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs

        self.pred_save_path = pred_save_path
        self.seed = seed
        self.hash = hash
        self.eval_last_epoch_only = eval_last_epoch_only

        self.val_names = val_names

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        images, _ = batch
        return self.model(images)

    def process_batch(self, batch):
        images, labels = batch
        logits = self.forward(images)
        probs = softmax(logits, dim=1)
        loss = self.criterion(logits, labels)
        accuracy = self.accuracy(probs, labels)
        return loss, accuracy * 100

    def batch_preds(self, batch):
        images, labels = batch
        logits = self.forward(images)
        preds = torch.argmax(logits, dim=1)
        preds_acc = preds == labels
        return preds_acc, labels

    def training_step(self, batch, batch_idx: int):
        loss, accuracy = self.process_batch(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def is_eval_epoch(self):
        return (not self.eval_last_epoch_only) or (
            self.eval_last_epoch_only and self.current_epoch == (self.max_epochs - 1)
        )

    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0):

        if dataloader_idx == 0:
            loss, accuracy = self.process_batch(batch)
            self.log("loss/val", loss, add_dataloader_idx=False)
            self.log("acc/val", accuracy, add_dataloader_idx=False)
        else:
            if self.is_eval_epoch():
                preds_acc, labels = self.batch_preds(batch)
                batch_info = pd.DataFrame(
                    {"acc": preds_acc.cpu().numpy(), "label": labels.cpu().numpy()}
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
            self.process_outputs(outputs)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=0.9,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )

        return [optimizer], [lr_scheduler]

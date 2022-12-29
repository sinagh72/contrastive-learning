import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics.functional import accuracy
from torchmetrics.functional.classification import multiclass_confusion_matrix


class LogisticRegression(pl.LightningModule):

    def __init__(self, feature_dim, classes, lr, weight_decay, max_epochs=100):
        """

        :param feature_dim:
        :param classes (tuple(str, int)): list of tuples, each tuple consists of class name and class index
        :param lr (float): learning rate
        :param weight_decay (float): weight decay of optimizer
        :param max_epochs (int): maximum epochs
        """
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, len(self.hparams.classes))
        self.train_cm = MulticlassConfusionMatrix(num_classes=len(self.hparams.classes))
        self.val_cm = MulticlassConfusionMatrix(num_classes=len(self.hparams.classes))
        self.test_cm = MulticlassConfusionMatrix(num_classes=len(self.hparams.classes))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs * 0.6),
                                                                  int(self.hparams.max_epochs * 0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        return {"loss": loss, "preds": torch.flatten(preds.argmax(dim=-1)), "labels": torch.flatten(labels)}

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.train_cm.update(preds, labels)
        return batch_parts["loss"]

    def training_epoch_end(self, outputs):
        cm = self.train_cm.compute()
        class_accuracy = 100 * cm.diagonal() / cm.sum(1)
        log = {}
        for c in self.hparams.classes:
            log["train_acc_" + c[0]] = class_accuracy[c[1]]
        log["train_loss"] = outputs[-1]

        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.train_cm.reset()

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def validation_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.val_cm.update(preds, labels)
        return batch_parts["loss"]

    def validation_epoch_end(self, outputs):
        cm = self.val_cm.compute()
        class_accuracy = 100 * cm.diagonal() / cm.sum(1)
        log = {}
        for c in self.hparams.classes:
            log["val_acc_" + c[0]] = class_accuracy[c[1]]
        log["val_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.val_cm.reset()

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def test_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.test_cm.update(preds, labels)
        return batch_parts["loss"]

    def test_epoch_end(self, outputs):
        cm = self.test_cm.compute()
        class_accuracy = 100 * cm.diagonal() / cm.sum(1)
        log = {}
        for c in self.hparams.classes:
            log["test_acc_" + c[0]] = class_accuracy[c[1]]
        log["test_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.test_cm.reset()

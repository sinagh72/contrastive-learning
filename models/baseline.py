import pytorch_lightning as pl
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy, MulticlassPrecision


class ResNet(pl.LightningModule):

    def __init__(self, classes, lr, weight_decay, metric="accuracy", max_epochs=100):
        """

        :param classes (tuple(str, int)): list of tuples, each tuple consists of class name and class index
        :param lr (float): learning rate
        :param weight_decay (float): weight decay of optimizer
        :param max_epochs (int): maximum epochs
        :param accuracy (str): specifies the type of metric
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(num_classes=len(self.hparams.classes))
        if self.hparams.metric == "accuracy":
            self.train_cm = MulticlassAccuracy(num_classes=len(self.hparams.classes), average=None)
            self.val_cm = MulticlassAccuracy(num_classes=len(self.hparams.classes), average=None)
            self.test_cm = MulticlassAccuracy(num_classes=len(self.hparams.classes), average=None)

        elif self.hparams.metric == "precision":
            self.train_cm = MulticlassPrecision(num_classes=len(self.hparams.classes), average=None)
            self.val_cm = MulticlassPrecision(num_classes=len(self.hparams.classes), average=None)
            self.test_cm = MulticlassPrecision(num_classes=len(self.hparams.classes), average=None)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs * 0.7),
                                                                  int(self.hparams.max_epochs * 0.9)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        return {"loss": loss, "preds": torch.flatten(preds.argmax(dim=-1)), "labels": torch.flatten(labels)}

    def _calculate_loss2(self, batch):
        all_outputs = self.all_gather(batch, sync_grads=True)
        feats, labels = all_outputs
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        return {"loss": loss, "preds": torch.flatten(preds.argmax(dim=-1)), "labels": torch.flatten(labels)}

    def training_step(self, batch, batch_idx):
        return self._calculate_loss2(batch)

    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.train_cm.update(preds, labels)
        return batch_parts["loss"]

    def training_epoch_end(self, outputs):
        cm = self.train_cm.compute()
        log = {}
        for c in self.hparams.classes:
            log[f"train_{self.hparams.metric}_" + c[0]] = cm[c[1]]
        log["train_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.train_cm.reset()

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss2(batch)

    def validation_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.val_cm.update(preds, labels)
        return batch_parts["loss"]

    def validation_epoch_end(self, outputs):
        cm = self.val_cm.compute()
        log = {}
        for c in self.hparams.classes:
            log[f"val_{self.hparams.metric}_" + c[0]] = cm[c[1]]
        log["val_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.val_cm.reset()

    def test_step(self, batch, batch_idx):
        return self._calculate_loss2(batch)

    def test_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.test_cm.update(preds, labels)
        return batch_parts["loss"]

    def test_epoch_end(self, outputs):
        cm = self.test_cm.compute()
        log = {}
        for c in self.hparams.classes:
            log[f"test_{self.hparams.metric}_" + c[0]] = cm[c[1]]
        log["test_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.test_cm.reset()

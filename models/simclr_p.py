from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.nn import functional as F
from torch.optim import Adam
from torchmetrics import F1Score
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision

class SimCLRP(pl.LightningModule):
    def __init__(self, encoder, freeze_num, feature_dim, classes, lr, weight_decay, metric="accuracy",
                 max_epochs=100):
        super().__init__()
        # self.model = encoder
        # SimCLR encoder
        self.classes = classes
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.metric = metric
        self.model = nn.Sequential(
            encoder,
            nn.Linear(feature_dim, 10*feature_dim),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(10*feature_dim, 5*feature_dim),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(5*feature_dim, len(classes))  # Linear(feature dim, #classes)
        )
        counter = 0
        for param in self.model.parameters():
            if counter == freeze_num:
                break
            param.requires_grad = False
            counter -= 1

        if self.metric == "accuracy":
            self.train_cm = MulticlassAccuracy(num_classes=len(self.classes), average=None)
            self.val_cm = MulticlassAccuracy(num_classes=len(self.classes), average=None)
            self.test_cm = MulticlassAccuracy(num_classes=len(self.classes), average=None)

        elif self.metric == "precision":
            self.train_cm = MulticlassPrecision(num_classes=len(self.classes), average=None)
            self.val_cm = MulticlassPrecision(num_classes=len(self.classes), average=None)
            self.test_cm = MulticlassPrecision(num_classes=len(self.classes), average=None)

        task = "binary" if len(self.classes) == 2 else "multiclass"
        self.train_f1 = F1Score(task=task, num_classes=3)
        self.val_f1 = F1Score(task=task, num_classes=len(self.classes))
        self.test_f1 = F1Score(task=task, num_classes=len(self.classes))

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.lr,
                                weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.max_epochs * 0.6),
                                                                  int(self.max_epochs * 0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch):
        imgs, labels = batch["img"], batch["label"]
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        return {"loss": loss, "preds": torch.flatten(preds.argmax(dim=-1)), "labels": torch.flatten(labels)}

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch)
    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.train_cm.update(preds, labels)
        self.train_f1.update(preds, labels)
        return batch_parts["loss"]

    def training_epoch_end(self, outputs):
        cm = self.train_cm.compute()
        # f1 = self.train_f1.compute()

        log = {}
        for c in self.classes:
            log[f"train_{self.metric}_" + c[0]] = cm[c[1]]

        # log["train_f1"] = f1
        log["train_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.train_cm.reset()
        self.train_f1.reset()

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def validation_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.val_cm.update(preds, labels)
        self.val_f1.update(preds, labels)
        return batch_parts["loss"]

    def validation_epoch_end(self, outputs):
        cm = self.val_cm.compute()
        f1 = self.val_f1.compute()
        log = {}
        for c in self.classes:
            log[f"val_{self.metric}_" + c[0]] = cm[c[1]]
        log["val_f1"] = f1
        log["val_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.val_cm.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def test_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        self.test_cm.update(preds, labels)
        self.test_f1.update(preds, labels)
        return batch_parts["loss"]

    def test_epoch_end(self, outputs):
        cm = self.test_cm.compute()
        f1 = self.test_f1.compute()
        log = {}
        for c in self.classes:
            log[f"test_{self.metric}_" + c[0]] = cm[c[1]]
        log["test_f1"] = f1
        log["test_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True)
        self.test_cm.reset()
        self.test_f1.reset()

import math

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.nn import functional as F
from torchmetrics import F1Score, AUROC
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision

class SimCLRP(pl.LightningModule):
    def __init__(self, encoder, freeze_p, feature_dim, classes, lr, weight_decay,
                 max_epochs=100):
        super().__init__()
        # self.model = encoder
        # SimCLR encoder
        self.classes = classes
        self.max_epochs = max_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        freeze_count = math.floor(sum(1 for _ in encoder.parameters()) * freeze_p)
        counter = 0
        for param in encoder.parameters():
            if freeze_count == counter:
                break
            param.requires_grad = False
            counter += 1

        self.model = nn.Sequential(
            encoder,
            nn.Linear(feature_dim, 10 * feature_dim),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(10 * feature_dim, 5 * feature_dim),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(5 * feature_dim, len(classes)),  # Linear(feature dim, #classes)
        )

        self.train_ac = MulticlassAccuracy(num_classes=len(self.classes), average=None)
        self.val_ac = MulticlassAccuracy(num_classes=len(self.classes), average=None)
        self.test_ac = MulticlassAccuracy(num_classes=len(self.classes), average=None)

        self.train_p = MulticlassPrecision(num_classes=len(self.classes), average=None)
        self.val_p = MulticlassPrecision(num_classes=len(self.classes), average=None)
        self.test_p = MulticlassPrecision(num_classes=len(self.classes), average=None)

        self.train_p = MulticlassPrecision(num_classes=len(self.classes), average=None)
        self.val_p = MulticlassPrecision(num_classes=len(self.classes), average=None)
        self.test_p = MulticlassPrecision(num_classes=len(self.classes), average=None)

        task = "binary" if len(self.classes) == 2 else "multiclass"
        self.train_f1 = F1Score(task=task, num_classes=3)
        self.val_f1 = F1Score(task=task, num_classes=len(self.classes))
        self.test_f1 = F1Score(task=task, num_classes=len(self.classes))

        self.train_auc = AUROC(task=task, num_classes=len(self.classes))
        self.val_auc = AUROC(task=task, num_classes=len(self.classes))
        self.test_auc = AUROC(task=task, num_classes=len(self.classes))

        self.metrics = {"train": [self.train_ac, self.train_p, self.train_f1,
                                  # self.train_auc
                                  ],
                        "val": [self.val_ac, self.val_p, self.val_f1,
                                # self.val_auc
                                ],
                        "test": [self.test_ac, self.test_p, self.test_f1,
                                 # self.test_auc
                                 ]}

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
        return {"loss": loss, "preds": torch.flatten(preds.argmax(dim=-1)), "labels": labels}

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def training_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        for metric in self.metrics["train"]:
            metric.update(preds, labels)
        self.train_auc.update(F.one_hot(preds, len(self.classes)).type(torch.float32).to(preds.get_device()), labels)
        return batch_parts["loss"]

    def training_epoch_end(self, outputs):
        cm = self.train_ac.compute()
        f1 = self.train_f1.compute()
        precision = self.train_p.compute()
        auc = self.train_auc.compute()

        log = {}
        for c in self.classes:
            log[f"train_accuracy_" + c[0]] = cm[c[1]]
            log[f"train_precision_" + c[0]] = precision[c[1]]

        log["train_f1"] = f1
        log["train_auc"] = auc
        log["train_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.metrics["train"]:
            metric.reset()
        self.train_auc.reset()

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def validation_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        for metric in self.metrics["val"]:
            metric.update(preds, labels)
        self.val_auc.update(F.one_hot(preds, len(self.classes)).type(torch.float32).to(preds.get_device()), labels)
        return batch_parts["loss"]

    def validation_epoch_end(self, outputs):
        cm = self.val_ac.compute()
        f1 = self.val_f1.compute()
        precision = self.val_p.compute()
        auc = self.val_auc.compute()

        log = {}
        for c in self.classes:
            log[f"val_accuracy_" + c[0]] = cm[c[1]]
            log[f"val_precision_" + c[0]] = precision[c[1]]

        log["val_f1"] = f1
        log["val_auc"] = auc
        log["val_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.metrics["val"]:
            metric.reset()
        self.val_auc.reset()

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch)

    def test_step_end(self, batch_parts):
        preds = batch_parts["preds"]
        labels = batch_parts["labels"]
        for metric in self.metrics["test"]:
            metric.update(preds, labels)
        self.test_auc.update(F.one_hot(preds, len(self.classes)).type(torch.float32).to(preds.get_device()), labels)
        return batch_parts["loss"]

    def test_epoch_end(self, outputs):
        cm = self.test_ac.compute()
        f1 = self.test_f1.compute()
        precision = self.test_p.compute()
        auc = self.test_auc.compute()

        log = {}
        for c in self.classes:
            log[f"test_accuracy_" + c[0]] = cm[c[1]]
            log[f"test_precision_" + c[0]] = precision[c[1]]

        log["test_f1"] = f1
        log["test_auc"] = auc
        log["test_loss"] = outputs[-1]
        self.log_dict(log, sync_dist=True, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.metrics["test"]:
            metric.reset()
        self.test_auc.reset()
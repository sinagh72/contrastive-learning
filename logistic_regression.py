import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchmetrics import ConfusionMatrix
from torchmetrics.functional import accuracy
from torchmetrics.functional.classification import multiclass_confusion_matrix


class LogisticRegression(pl.LightningModule):

    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs * 0.6),
                                                                  int(self.hparams.max_epochs * 0.8)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        print(preds.argmax(dim=-1).shape)
        print(labels.shape)
        out = accuracy(preds=preds.argmax(dim=-1).to(torch.int), target=labels.to(torch.int),
                       task="multiclass", num_classes=3)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        log_dict = {mode + "_confusion": out,
                    mode + "_loss": loss,
                    mode + "_acc": acc}
        return log_dict

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def training_step_end(self, batch_parts):
        # losses from each GPU
        log_dict = {}
        # do something with both outputs
        for k, v in batch_parts.items():
            log_dict[k] = batch_parts[k].mean()

        self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
        return batch_parts["train_loss"].mean()

    def validation_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='val')

    def validation_step_end(self, batch_parts):
        log_dict = {}
        # do something with both outputs
        for k, v in batch_parts.items():
            log_dict[k] = batch_parts[k].mean()

        self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
        return batch_parts["val_loss"].mean()

    def test_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='test')

    def test_step_end(self, batch_parts):
        # losses from each GPU
        log_dict = {}
        # do something with both outputs
        for k, v in batch_parts.items():
            log_dict[k] = batch_parts[k].mean()

        self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
        return batch_parts["test_loss"].mean()

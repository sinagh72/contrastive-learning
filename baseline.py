import pytorch_lightning as pl
import torchvision
from torch import optim
import torch.nn.functional as F


class ResNet(pl.LightningModule):

    def __init__(self, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                      milestones=[int(self.hparams.max_epochs * 0.7),
                                                                  int(self.hparams.max_epochs * 0.9)],
                                                      gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        imgs, labels = batch["img"], batch["y_true"]
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        log_dict = {mode + "_loss": loss,
                    mode + "_acc": acc}
        # self.log(mode + '_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # self.log(mode + '_acc', acc, on_epoch=True, prog_bar=True, sync_dist=True)
        # return  loss
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

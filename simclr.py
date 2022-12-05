import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from torch import optim
from torch.nn import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500, n_views=2):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(num_classes=4 * hidden_dim)  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(),
                                lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.hparams.max_epochs,
                                                            eta_min=self.hparams.lr / 50)
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode='train'):
        # list of augmentation list
        imgs = batch["img"]
        # imgs is a tensor of (B*n_view, 3, H, W), for the batch=64 and n_view=2 we would have (128, 3, 128, 128)
        imgs = torch.cat(imgs, dim=0)
        # print(self.device, len(imgs))
        # img_grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, pad_value=0.9).cpu()
        # img_grid = img_grid.permute(1, 2, 0)
        #
        # plt.figure(figsize=(10,5))
        # plt.title('Augmented image examples of the STL10 dataset')
        # plt.imshow(img_grid)
        # plt.axis('off')
        # plt.show()
        # plt.close()
        # Encode all images (B*n_view, hidden_dim), hidden_dim=128 in this case
        feats = self.convnet(imgs)
        # Calculate cosine similarity between all images in the batch
        # (128, 1, 128) and (1, 128, 128) --> z.z' = (128, 128) the table relation of images
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//n_views away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // self.hparams.n_views, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        # -log( exp(sim(zi,zj)/t) / sum(exp(sim(zi,zk)/t)) )
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        # self.log(mode + '_loss', nll, sync_dist=True)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        log_dict = {mode + '_loss': nll,
                    #mode + '_acc_top1': (sim_argsort == 0).float().mean(),
                    #mode + '_acc_top5': (sim_argsort < 5).float().mean(),
                    #mode + '_acc_mean_pos': 1 + sim_argsort.float().mean()
        }

        # self.log_dict(log_dict, sync_dist=True)
        # self.log(mode + '_acc_top5', (sim_argsort < 5).float().mean(), sync_dist=True)
        # self.log(mode + '_acc_mean_pos', 1 + sim_argsort.float().mean(), sync_dist=True)
        return log_dict

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='train')

    def training_step_end(self, batch_parts):
        # losses from each GPU
        log_dict = {}
        # do something with both outputs
        for k, v in batch_parts.items():
            log_dict[k] = batch_parts[k].mean()

        self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
        return batch_parts["train_loss"].mean()
    # def training_epoch_end(self, training_step_outputs):
    #     loss = torch.stack([x for x in training_step_outputs]).mean()
    #     log_dict = {"train_loss": loss}
    #     self.log_dict(log_dict, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode='val')

    def validation_step_end(self, batch_parts):
        log_dict = {}
        # do something with both outputs
        for k, v in batch_parts.items():
            log_dict[k] = batch_parts[k].mean()

        self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
        return batch_parts["val_loss"].mean()
        # def validation_epoch_end(self, validation_step_outputs):
    #     loss = torch.stack([x for x in validation_step_outputs]).mean()
    #     log_dict = {"val_loss": loss}
    #     self.log_dict(log_dict, prog_bar=True)

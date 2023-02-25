import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from torch import optim
from torch.nn import functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.adam import Adam


class SimCLR(pl.LightningModule):

    def __init__(self, gpus: int = 0, batch_size: int = 450, num_samples: int = 0, hidden_dim: int = 2048,
                 feature_dim: int = 128, lr: float = 1e-3, temperature: float = 0.1, warmup_epochs: int = 10,
                 weight_decay: float = 1e-6, max_epochs: int = 500, n_views: int = 2,
                 gradient_accumulation_steps: int = 5):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        # Output of last linear layer
        self.convnet = torchvision.models.resnet18(num_classes=hidden_dim)
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, feature_dim)
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False)  # Linear(hidden_dim, feature_dim)
        )
        # compute iters per epoch
        # global_batch_size = gpus * batch_size if gpus > 0 else batch_size
        # self.train_iters_per_epoch = num_samples // global_batch_size

    def forward(self, x):
        return self.convnet(x)

    def configure_optimizers(self):
        max_epochs = self.hparams.max_epochs
        param_groups = define_param_groups(self.convnet, self.hparams.weight_decay, 'adam')
        optimizer = Adam(param_groups, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

        print(f'Optimizer Adam, '
              f'Learning Rate {self.hparams.lr}, '
              f'Effective batch size {self.hparams.batch_size * self.hparams.gradient_accumulation_steps}')

        scheduler_warmup = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=max_epochs,
                                                         warmup_start_lr=0.0)

        return [optimizer], [scheduler_warmup]

    def info_nce_loss(self, feats):
        # Calculate cosine similarity between all images in the batch
        # (batch_size*n_views, 1, hidden_dim) and (1, batch_size*n_views, hidden_dim)
        # --> z.z' = (batch_size*n_views, batch_size*n_views) the table relation of images
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
        # nll = nll.mean()

        # Logging loss
        # self.log(mode + '_loss', nll, sync_dist=True)
        # Get ranking position of positive example
        comb_sim = torch.cat([cos_sim[pos_mask][:, None],  # First position positive example
                              cos_sim.masked_fill(pos_mask, -9e15)],
                             dim=-1)
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)

        return nll, sim_argsort

    def compute_loss(self, batch, mode='train'):
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
        nll, sim_argsort = self.info_nce_loss(feats)
        nll = nll.mean()
        # Logging ranking metrics
        log_dict = {mode + '_loss': nll,
                    mode + '_acc_top1': (sim_argsort == 0).float().mean(),
                    mode + '_acc_top10': (sim_argsort < 10).float().mean(),
                    mode + '_acc_mean_pos': sim_argsort.float().mean()
                    }

        # self.log_dict(log_dict, sync_dist=True)
        # self.log(mode + '_acc_top5', (sim_argsort < 5).float().mean(), sync_dist=True)
        # self.log(mode + '_acc_mean_pos', 1 + sim_argsort.float().mean(), sync_dist=True)
        # count = Batch_size * n_views
        # counts = 1.0 if nll.numel() == 0 else nll.size(dim=0)
        # log_dict = {"loss": nll.sum(), "count": float(counts)}
        # return log_dict
        self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
        return nll

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch, mode='train')

    # def training_step_end(self, batch_parts):
    #     # # do something with both outputs
    #     # for k, v in batch_parts.items():
    #     #     log_dict[k] = batch_parts[k].mean()
    #     nll = batch_parts["loss"]
    #     count = batch_parts["count"]
    #     log_dict = {"train_loss_step": torch.div(nll, count)}
    #     # log_dict = {"train_loss_step": batch_parts["loss"]}
    #     self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
    #     return log_dict["train_loss_step"]

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, mode='val')

    # def validation_step_end(self, batch_parts):
    #     # log_dict = {}
    #     # do something with both outputs
    #     # for k, v in batch_parts.items():
    #     # log_dict[k] = batch_parts[k].mean()
    #
    #     # self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
    #     # return batch_parts["val_loss"].mean()
    #     # def validation_epoch_end(self, validation_step_outputs):
    #     #     loss = torch.stack([x for x in validation_step_outputs]).mean()
    #     #     log_dict = {"val_loss": loss}
    #     #     self.log_dict(log_dict, prog_bar=True)
    #     nll = batch_parts["loss"]
    #     count = batch_parts["count"]
    #     log_dict = {"val_loss": torch.div(nll, count)}
    #     # log_dict = {"val_loss": batch_parts["loss"]}
    #     self.log_dict(log_dict, prog_bar=True, on_step=True, sync_dist=True)
    #     return log_dict["val_loss"]


def define_param_groups(model, weight_decay, optimizer_name):
    def exclude_from_wd_and_adaptation(name):
        if 'bn' in name:
            return True
        if optimizer_name == 'lars' and 'bias' in name:
            return True

    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': weight_decay,
            'layer_adaptation': True,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'layer_adaptation': False,
        },
    ]
    return param_groups

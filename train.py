import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from torch.utils.data import DataLoader

from simclr import SimCLR

NUM_WORKERS = 8


def train_simclr(batch_size, max_epochs=500, train_data=None, val_data=None, checkpoint_path=None, **kwargs):
    # Setting the seed
    pl.seed_everything(42)

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, 'SimCLR'),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=4,
                         strategy="ddp",
                         log_every_n_steps=8,
                         num_nodes=1,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc_top5'),
                                    LearningRateMonitor('epoch')])
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path, 'SimCLR.ckpt')
    if os.path.isfile(pretrained_filename):
        print(f'Found pretrained model at {pretrained_filename}, loading...')
        model = SimCLR.load_from_checkpoint(
            pretrained_filename)  # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        model = SimCLR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path)  # Load best checkpoint after training

    return model

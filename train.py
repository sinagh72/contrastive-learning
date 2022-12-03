import os
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from baseline import ResNet
from logistic_regression import LogisticRegression
from simclr import SimCLR

NUM_WORKERS = os.cpu_count()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
devices = torch.cuda.device_count()
# devices = 5
strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)


def train_simclr(batch_size, max_epochs=500, train_data=None, val_data=None, checkpoint_path=None, save_model_name=None,
                 **kwargs):
    pl.seed_everything(42)
    model_path = os.path.join(checkpoint_path, save_model_name)
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[
                             ModelCheckpoint(dirpath=model_path, filename=save_model_name,
                                             save_weights_only=True, mode='max', monitor='val_acc_top5'),
                             LearningRateMonitor('epoch')],
                         log_every_n_steps=1)
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(model_path):
        print(f'Found pretrained model at {model_path}, loading...')
        model = SimCLR.load_from_checkpoint(model_path)  # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                  drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


def train_logreg(batch_size, train_feats_data, test_feats_data, checkpoint_path, log_every_n_steps, max_epochs=100,
                 save_model_name=None, **kwargs):
    model_path = os.path.join(checkpoint_path, save_model_name)
    trainer = pl.Trainer(default_root_dir=save_model_name,
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(dirpath=model_path, filename=save_model_name,
                                                    save_weights_only=True, mode='max', monitor='val_acc'),
                                    LearningRateMonitor("epoch")],
                         log_every_n_steps=log_every_n_steps)
    # trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    # Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(model_path):
        print(f"Found pretrained model at {model_path}, loading...")
        model = LogisticRegression.load_from_checkpoint(model_path)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result


@torch.no_grad()
def prepare_data_features(model, dataset, batch_size=64):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch in tqdm(data_loader):
        batch_imgs = batch["img"].to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch["y_true"])

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    #
    # # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return TensorDataset(feats, labels)


def train_resnet(batch_size, train_data, test_data, checkpoint_path, log_every_n_steps, max_epochs=100,
                 save_model_name=None, **kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(checkpoint_path, save_model_name),
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         log_every_n_steps=log_every_n_steps)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(checkpoint_path, save_model_name+".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        model = ResNet.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = ResNet(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": val_result[0]["test_acc"]}

    return model, result

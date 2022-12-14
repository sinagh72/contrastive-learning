import os
from copy import deepcopy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.strategies.ddp import DDPStrategy
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from baseline import ResNet
from logistic_regression import LogisticRegression
from simclr import SimCLR

NUM_WORKERS = os.cpu_count()


def train_simclr(batch_size, max_epochs=500, train_data=None, val_data=None, checkpoint_path=None, save_model_name=None,
                 devices=1, strategy=None, **kwargs):
    pl.seed_everything(42)
    model_path = os.path.join(checkpoint_path, save_model_name)
    early_stopping = EarlyStopping(monitor="train_loss", patience=50, verbose=False, mode="min")
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[
                             early_stopping,
                             ModelCheckpoint(dirpath=model_path, filename=save_model_name,
                                             save_weights_only=True, mode='min', monitor='train_loss'),
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

        model = SimCLR(max_epochs=max_epochs, **kwargs)
        if val_data:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


def train_logreg(batch_size, train_feats_data, val_feats_data, test_feats_data, checkpoint_path, max_epochs=100,
                 save_model_name=None, devices=1, strategy=None, **kwargs):
    model_path = os.path.join(checkpoint_path, save_model_name)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[early_stopping,
                                    ModelCheckpoint(dirpath=model_path, filename=save_model_name,
                                                    save_weights_only=True, mode='min', monitor='val_loss'),
                                    LearningRateMonitor("epoch")],
                         log_every_n_steps=1)
    trainer.logger._default_hp_metric = None
    # Data loaders
    train_loader = DataLoader(train_feats_data, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_feats_data, batch_size=batch_size, shuffle=False,
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
        trainer.fit(model, train_loader, val_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)

    result = {"train": {"acc_normal": train_result[0]["test_acc_normal"],
                        "acc_ADM": train_result[0]["test_acc_AMD"],
                        "acc_DME": train_result[0]["test_acc_DME"]},
              "val": {"acc_normal": val_result[0]["test_acc_normal"],
                      "acc_ADM": val_result[0]["test_acc_AMD"],
                      "acc_DME": val_result[0]["test_acc_DME"]},
              "test": {"acc_normal": test_result[0]["test_acc_normal"],
                       "acc_ADM": test_result[0]["test_acc_AMD"],
                       "acc_DME": test_result[0]["test_acc_DME"]}
              }
    return model, result


@torch.no_grad()
def prepare_data_features(model, dataset, device, batch_size=64):
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


def train_resnet(batch_size, train_data, val_data, test_data, checkpoint_path, max_epochs=100,
                 save_model_name=None, devices=1, strategy=None, **kwargs):
    model_path = os.path.join(checkpoint_path, save_model_name)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[early_stopping,
                                    ModelCheckpoint(
                                        dirpath=model_path, filename=save_model_name, save_weights_only=True, mode="min"
                                        , monitor="val_loss"),
                                    LearningRateMonitor("epoch")],
                         log_every_n_steps=1)
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    # Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(model_path):
        print("Found pretrained model at %s, loading..." % model_path)
        model = ResNet.load_from_checkpoint(model_path)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = ResNet(**kwargs)
        trainer.fit(model, train_loader, valid_loader)
        model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, valid_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)

    result = {"train": {"acc_normal": train_result[0]["test_acc_normal"],
                        "acc_ADM": train_result[0]["test_acc_AMD"],
                        "acc_DME": train_result[0]["test_acc_DME"]},
              "val": {"acc_normal": val_result[0]["test_acc_normal"],
                      "acc_ADM": val_result[0]["test_acc_AMD"],
                      "acc_DME": val_result[0]["test_acc_DME"]},
              "test": {"acc_normal": test_result[0]["test_acc_normal"],
                       "acc_ADM": test_result[0]["test_acc_AMD"],
                       "acc_DME": test_result[0]["test_acc_DME"]}
              }

    return model, result

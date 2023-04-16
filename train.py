import os
from copy import deepcopy
from pytorch_lightning import loggers as pl_loggers
import torch

from models.simclr_p import SimCLRP

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, \
    GradientAccumulationScheduler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.baseline import ResNet
from models.linear_model import LinearModel
from models.simclr import SimCLR

NUM_WORKERS = os.cpu_count()


def train_simclr(batch_size, max_epochs=500, train_data=None, val_data=None, checkpoint_path=None, save_model_name=None,
                 devices=1, strategy=None, monitor="", mode="min", patience=15, **kwargs):
    pl.seed_everything(42)
    model_path = os.path.join(checkpoint_path, save_model_name)
    early_stopping = EarlyStopping(monitor=monitor, patience=patience, verbose=False, mode=mode)
    accumulator = GradientAccumulationScheduler(scheduling={0: kwargs["gradient_accumulation_steps"]})
    tb_logger = pl_loggers.CSVLogger(save_dir=os.path.join(model_path, "log/"))
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=max_epochs,
                         callbacks=[
                             accumulator,
                             early_stopping,
                             ModelCheckpoint(dirpath=model_path, filename=save_model_name, save_top_k=1,
                                             save_weights_only=True, mode=mode, monitor=monitor),
                             LearningRateMonitor('epoch')],
                         logger=tb_logger,
                         log_every_n_steps=1,
                         num_nodes=1,
                         sync_batchnorm=True
                         )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don'resent need

    # Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(model_path):
        print(f'Found pretrained model at {model_path}, loading...')
        model = SimCLR.load_from_checkpoint(model_path)  # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, persistent_workers=True,
                                  drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

        model = SimCLR(max_epochs=max_epochs, batch_size=batch_size, **kwargs)
        if val_data:
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, persistent_workers=True,
                                    drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)
        # Load best checkpoint after training
    #     model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    #
    # return model


def train_linear_model(batch_size, train_feats_data, val_feats_data, test_feats_data, checkpoint_path,
                       save_model_name=None, devices=1, strategy=None, **kwargs):
    metric = "accuracy"
    if "metric" in kwargs:
        metric = kwargs["metric"]

    model_path = os.path.join(checkpoint_path, save_model_name)
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=False, mode="min")
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=kwargs["max_epochs"],
                         callbacks=[early_stopping,
                                    ModelCheckpoint(dirpath=model_path, filename=save_model_name,
                                                    save_weights_only=True, mode='min', monitor='val_loss',
                                                    save_top_k=1),
                                    LearningRateMonitor("epoch")],
                         log_every_n_steps=1,
                         sync_batchnorm=True)
    trainer.logger._default_hp_metric = None
    # Data loaders
    train_loader = DataLoader(train_feats_data, batch_size=batch_size, shuffle=True, persistent_workers=True,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)

    val_loader = DataLoader(val_feats_data, batch_size=batch_size, shuffle=False, persistent_workers=True,
                            drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    test_loader = DataLoader(test_feats_data, batch_size=batch_size, shuffle=False, persistent_workers=True,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    # Check whether pretrained model exists. If yes, load it and skip training
    if os.path.isfile(model_path):
        print(f"Found pretrained model at {model_path}, loading...")
        model = LinearModel.load_from_checkpoint(model_path)
    else:
        pl.seed_everything(42)  # To be reproducable
        model = LinearModel(**kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = LinearModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": {}, "val": {}, "test": {}}
    for c in kwargs["classes"]:
        result["train"][f"{metric}_" + c[0]] = train_result[0][f"test_{metric}_" + c[0]]
        result["val"][f"{metric}_" + c[0]] = val_result[0][f"test_{metric}_" + c[0]]
        result["test"][f"{metric}_" + c[0]] = test_result[0][f"test_{metric}_" + c[0]]

    return model, result


@torch.no_grad()
def prepare_data_features(model, dataset, device, num_workers, batch_size=64):
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
        labels.append(batch["label"])

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    #
    # # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return TensorDataset(feats, labels)


def train_resnet(batch_size, train_data, val_data, test_data, checkpoint_path,
                 save_model_name=None, devices=1, strategy=None, monitor="val_loss", patience=10, mode="max", **kwargs):
    model_path = os.path.join(checkpoint_path, save_model_name)

    callbacks = [LearningRateMonitor("epoch")]
    if monitor is not None:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, verbose=False,
                                       mode=mode))
        callbacks.append(ModelCheckpoint(dirpath=model_path, filename=save_model_name, save_weights_only=True,
                                         mode=mode, monitor=monitor, save_top_k=1))

    tb_logger = pl_loggers.CSVLogger(save_dir=os.path.join(model_path, "log/"))
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=kwargs["max_epochs"],
                         callbacks=callbacks,
                         log_every_n_steps=1,
                         sync_batchnorm=True,
                         num_nodes=1,
                         logger=tb_logger)
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
    result = {"train": {}, "val": {}, "test": {}}
    # metrics = ["accuracy", "precision"]
    # for c in kwargs["classes"]:
    #     for m in metrics:
    #         result["train"][f"{m}_" + c[0]] = train_result[0][f"test_{m}_" + c[0]]
    #         result["val"][f"{m}_" + c[0]] = val_result[0][f"test_{m}_" + c[0]]
    #         result["test"][f"{m}_" + c[0]] = test_result[0][f"test_{m}_" + c[0]]
    result["test"]["f1"] = test_result[0]["test_f1"]
    result["val"]["f1"] = val_result[0]["test_f1"]
    result["train"]["f1"] = train_result[0]["test_f1"]

    result["test"]["auc"] = test_result[0]["test_auc"]
    result["val"]["auc"] = val_result[0]["test_auc"]
    result["train"]["auc"] = train_result[0]["test_auc"]

    result["test"]["loss"] = train_result[0]["test_loss"]

    return model, result


def train_simclr_p(batch_size, train_dataset, val_dataset, test_dataset, checkpoint_path,
                   save_model_name=None, devices=1, strategy=None, mode="min", monitor="val_loss", patience=10,
                   encoder_path="SimCLR",
                   **kwargs):
    pl.seed_everything(42)  # To be reproducable

    model_path = os.path.join(checkpoint_path, save_model_name)
    callbacks = [LearningRateMonitor("epoch")]
    if monitor is not None:
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, verbose=False,
                                       mode=mode))
        callbacks.append(ModelCheckpoint(dirpath=model_path, filename=save_model_name, save_weights_only=True,
                                         mode=mode, monitor=monitor, save_top_k=1))

    tb_logger = pl_loggers.CSVLogger(save_dir=os.path.join(model_path, "log/"))
    trainer = pl.Trainer(default_root_dir=model_path,
                         accelerator="gpu",
                         devices=devices,
                         strategy=strategy,
                         max_epochs=kwargs["max_epochs"],
                         callbacks=callbacks,
                         log_every_n_steps=1,
                         num_nodes=1,
                         sync_batchnorm=True,
                         logger=tb_logger)
    trainer.logger._default_hp_metric = None

    simclr_model = SimCLR.load_from_checkpoint(
        os.path.join(checkpoint_path, "SimCLR", encoder_path, encoder_path + ".ckpt"))

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)

    model = SimCLRP(encoder=simclr_model, **kwargs)
    trainer.fit(model, train_loader, valid_loader)
    model = SimCLRP.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                         encoder=model.encoder, **kwargs)

    # Test best model on validation set
    train_result = trainer.test(model, train_loader, verbose=False)
    val_result = trainer.test(model, valid_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"train": {}, "val": {}, "test": {}}
    # metrics = ["accuracy", "precision"]
    # for c in kwargs["classes"]:
    #     for m in metrics:
    #         result["train"][f"{m}_" + c[0]] = train_result[0][f"test_{m}_" + c[0]]
    #         result["val"][f"{m}_" + c[0]] = val_result[0][f"test_{m}_" + c[0]]
    #         result["test"][f"{m}_" + c[0]] = test_result[0][f"test_{m}_" + c[0]]
    result["test"]["f1"] = test_result[0]["test_f1"]
    result["val"]["f1"] = val_result[0]["test_f1"]
    result["train"]["f1"] = train_result[0]["test_f1"]

    result["test"]["auc"] = test_result[0]["test_auc"]
    result["val"]["auc"] = val_result[0]["test_auc"]
    result["train"]["auc"] = train_result[0]["test_auc"]

    result["test"]["loss"] = train_result[0]["test_loss"]

    return model, result

# def train_new_simclr(batch_size, max_epochs=500, train_data=None, val_data=None, checkpoint_path=None,
#                      save_model_name=None, devices=1, strategy=None, monitor="", mode="min", **kwargs):
#     pl.seed_everything(42)
#     model_path = os.path.join(checkpoint_path, save_model_name)
#     early_stopping = EarlyStopping(monitor=monitor, patience=50, verbose=False, mode=mode)
#     logger = TensorBoardLogger(model_path, name=save_model_name + "_tensor_board")
#     trainer = pl.Trainer(default_root_dir=model_path,
#                          accelerator="gpu",
#                          devices=devices,
#                          strategy=strategy,
#                          max_epochs=max_epochs,
#                          callbacks=[
#                              early_stopping,
#                              ModelCheckpoint(dirpath=model_path, filename=save_model_name,
#                                              save_weights_only=True, mode=mode, monitor=monitor),
#                              LearningRateMonitor('epoch')],
#                          logger=logger,
#                          log_every_n_steps=1,
#                          sync_batchnorm=True)
#     trainer.logger._default_hp_metric = None  # Optional logging argument that we don'resent need
#
#     # Check whether pretrained model exists. If yes, load it and skip training
#     if os.path.isfile(model_path):
#         print(f'Found pretrained model at {model_path}, loading...')
#         model = SimCLR.load_from_checkpoint(model_path)  # Automatically loads the model with the saved hyperparameters
#     else:
#         train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
#                                   drop_last=True, pin_memory=True, num_workers=NUM_WORKERS)
#         weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet' \
#                       ' / simclr_imagenet.ckpt'
#         simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
#         model = simclr_module.SimCLR(gpus=devices, nodes=1, batch_size=batch_size)
#         if val_data:
#             val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
#                                     drop_last=False, pin_memory=True, num_workers=NUM_WORKERS)
#
#             trainer.fit(model, train_loader, val_loader)
#         else:
#             trainer.fit(model, train_loader)
#         # Load best checkpoint after training
#         model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
#
#     return model

import numpy as np
from pytorch_lightning.strategies import DDPStrategy
from torchvision.transforms import transforms, InterpolationMode
import os
import torch

from OCT_dataset import DRDataset
from models.simclr import SimCLR
from train import prepare_data_features, train_resnet, train_linear_model


def _to_three_channel(x):
    return torch.cat([x, x, x], 0)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    devices = torch.cuda.device_count()
    devices = 1
    N_VIEWS = 2
    CV = 5
    # Path to the folder where the datasets are
    DATASET_PATH = "data/DR data-reorganized"
    # Path to load simclr and to save resnet and linear models
    CHECKPOINT_PATH = "./dr_saved_models/"
    # In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
    # workers as possible in a data loader, which corresponds to the number of CPU cores
    NUM_WORKERS = os.cpu_count()

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)

    n_views = 2

    img_transforms = transforms.Compose([transforms.Resize(size=(128, 128), interpolation=InterpolationMode.BICUBIC),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,)),
                                         transforms.Lambda(_to_three_channel)])
    classes = [("nonreferral", 0),
               ("referral", 1)]
    metric = "precision"
    batch_sizes = [1, 2, 5, 10]

    for batch_size in batch_sizes:
        for i in range(CV):
            train_dataset = DRDataset(data_root=DATASET_PATH + "/train",
                                      transform=img_transforms,
                                      classes=classes,
                                      mode="train",
                                      cv=CV,
                                      cv_counter=i
                                      )
            val_dataset = DRDataset(data_root=DATASET_PATH + "/train",
                                    transform=img_transforms,
                                    classes=classes,
                                    mode="val",
                                    cv=CV,
                                    cv_counter=i
                                    )

            test_dataset = DRDataset(data_root=DATASET_PATH + "/test",
                                     transform=img_transforms,
                                     classes=classes,
                                     mode="test",
                                     )

            simclr_model = SimCLR.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR", "SimCLR_" + str(i),
                                                                    "SimCLR_" + str(i) + ".ckpt"))
            print("training data preparation")
            train_feats_simclr = prepare_data_features(model=simclr_model,
                                                       dataset=train_dataset,
                                                       device=device,
                                                       batch_size=batch_size,
                                                       num_workers=1)
            print("validation data preparation")
            val_feats_simclr = prepare_data_features(model=simclr_model,
                                                     dataset=val_dataset,
                                                     device=device,
                                                     batch_size=batch_size,
                                                     num_workers=1)
            print("testing data preparation")
            test_feats_simclr = prepare_data_features(model=simclr_model,
                                                      dataset=test_dataset,
                                                      device=device,
                                                      batch_size=batch_size,
                                                      num_workers=1)

            strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
            lmodel_model, lmodel_result = train_linear_model(devices=devices,
                                                             strategy=strategy,
                                                             batch_size=batch_size,
                                                             train_feats_data=train_feats_simclr,
                                                             val_feats_data=val_feats_simclr,
                                                             test_feats_data=test_feats_simclr,
                                                             feature_dim=train_feats_simclr.tensors[0].shape[1],
                                                             classes=classes,
                                                             checkpoint_path=CHECKPOINT_PATH + "LinearModel",
                                                             lr=1e-3,
                                                             weight_decay=1e-3,
                                                             max_epochs=100,
                                                             metric=metric,
                                                             save_model_name="LinearModel" + str(i))

            file_mode = "a" if os.path.exists(f'log/dr_{metric}_lmodel_{batch_size}.txt') else "w"
            with open(f'log/dr_{metric}_lmodel_{batch_size}.txt', file_mode) as f:
                f.write("==================" + str(i) + "==================")
                f.write('\n')
                f.write(str(lmodel_result['train']))
                f.write('\n' + str(lmodel_result['val']))
                f.write('\n' + str(lmodel_result['test']))
                f.write('\n')

            strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
            resnet_model, resnet_result = train_resnet(devices=devices,
                                                       strategy=strategy,
                                                       batch_size=batch_size,
                                                       train_data=train_dataset,
                                                       val_data=val_dataset,
                                                       test_data=test_dataset,
                                                       lr=1e-3,
                                                       weight_decay=2e-4,
                                                       checkpoint_path=CHECKPOINT_PATH + "/ResNet",
                                                       max_epochs=100,
                                                       classes=classes,
                                                       metric=metric,
                                                       save_model_name="ResNet" + str(i))

            file_mode = "a" if os.path.exists(f'log/dr_{metric}_resnet_{batch_size}.txt') else "w"
            with open(f'log/dr_{metric}_resnet_{batch_size}.txt', file_mode) as f:
                f.write("==================" + str(i) + "==================")
                f.write('\n')
                f.write(str(resnet_result['train']))
                f.write('\n' + str(resnet_result['val']))
                f.write('\n' + str(resnet_result['test']))
                f.write('\n')


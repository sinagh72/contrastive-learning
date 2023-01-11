import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from pytorch_lightning.strategies import DDPStrategy

from OCT_dataset import OCTDataset, ContrastiveTransformations, train_aug, KaggleOCTDataset, DRDataset
from train import train_simclr

plt.set_cmap('cividis')
import matplotlib

matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns

sns.set()
import torch

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    devices = torch.cuda.device_count()
    devices = 1
    N_VIEWS = 2
    CV = 5
    # Path to the folder where the datasets are
    DATASET_PATH = "data/DR data-reorganized/train"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./dr_saved_models/SimCLR/"
    # In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
    # workers as possible in a data loader, which corresponds to the number of CPU cores
    NUM_WORKERS = os.cpu_count()

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)

    classes = [("nonreferral", 0),
               ("referral", 1)]
    for i in range(CV):
        train_dataset = DRDataset(data_root=DATASET_PATH,
                                  transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
                                  classes=classes,
                                  mode="train",
                                  cv=CV,
                                  cv_counter=i
                                  )
        val_dataset = DRDataset(data_root=DATASET_PATH,
                                transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
                                classes=classes,
                                mode="val",
                                cv=CV,
                                cv_counter=i
                                )
        print("len train:", len(train_dataset))
        print("len val: ", len(val_dataset))
        strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
        simclr_model = train_simclr(devices=devices,
                                    strategy=strategy,
                                    batch_size=min(len(train_dataset) // devices, 450),
                                    # batch_size=100,
                                    max_epochs=2000,
                                    train_data=train_dataset,
                                    val_data=val_dataset,
                                    checkpoint_path=CHECKPOINT_PATH,
                                    hidden_dim=128,
                                    lr=5e-4,
                                    temperature=0.07,
                                    weight_decay=1e-4,
                                    n_views=N_VIEWS,
                                    save_model_name="SimCLR_" + str(i),
                                    monitor="val_loss_epoch",
                                    mode="min"
                                    )

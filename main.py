import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from pytorch_lightning.strategies import DDPStrategy

from OCT_dataset import OCTDataset, ContrastiveTransformations, train_aug
from train import train_simclr

plt.set_cmap('cividis')
import matplotlib

matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns

sns.set()
import torch


def show_img(train, num_imgs=6, n_views=2):
    imgs = torch.stack([img for idx in range(num_imgs) for img in train[idx]["img"]], dim=0)
    img_grid = torchvision.utils.make_grid(imgs, nrow=n_views, normalize=True, pad_value=0.9)
    img_grid = img_grid.permute(1, 2, 0)

    plt.figure(figsize=(10, 5))
    plt.title('Augmented image examples of the 2014 BOE Srinivasan2 dataset')
    plt.imshow(img_grid)
    plt.axis('off')
    plt.show()
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    devices = torch.cuda.device_count()
    devices = 8
    N_VIEWS = 2
    CV = 5
    PATIENTS = 15
    cv_step = PATIENTS // CV
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "data/2014_BOE_Srinivasan_2/Publication_Dataset/original data"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./saved_models/SimCLR/"
    # In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
    # workers as possible in a data loader, which corresponds to the number of CPU cores
    NUM_WORKERS = os.cpu_count()

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)

    idx = np.array(range(1, cv_step + 1))

    for i in range(CV):
        train_dataset = OCTDataset(data_root=DATASET_PATH,
                                   img_suffix='.tif',
                                   transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
                                   folders=idx,
                                   extra_folder_names="TIFFs/8bitTIFFs")
        # print(set(np.array(range(1, PATIENTS + 1))) -set(choices))
        # val_dataset = OCTDataset(data_root=DATASET_PATH,
        #                          img_suffix='.tif',
        #                          transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
        #                          folders=list(set(np.array(range(1, PATIENTS + 1))) - set(idx)),
        #                          extra_folder_names="TIFFs/8bitTIFFs")

        print("len train:", len(train_dataset))
        # print("len val: ", len(val_dataset))
        strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
        simclr_model = train_simclr(devices=devices,
                                    strategy=strategy,
                                    batch_size=len(train_dataset) // devices,
                                    # batch_size=100,
                                    max_epochs=2000,
                                    train_data=train_dataset,
                                    # val_data=val_dataset,
                                    checkpoint_path=CHECKPOINT_PATH,
                                    hidden_dim=128,
                                    lr=5e-4,
                                    temperature=0.07,
                                    weight_decay=1e-4,
                                    n_views=N_VIEWS,
                                    save_model_name="SimCLR_" + str(idx))

        idx += cv_step

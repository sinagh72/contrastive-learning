import os
import matplotlib.pyplot as plt
import torchvision
from custom_dataset import CustomDataset, ContrastiveTransformations, train_aug
from train import train_simclr, train_resnet
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
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "./2014_BOE_Srinivasan_2/Publication_Dataset"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./saved_models"
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

    train_dataset_contrastive = CustomDataset(data_root=DATASET_PATH + "/Train", mode="train", img_suffix='.tif',
                                              transform=ContrastiveTransformations(train_aug, n_views=n_views))

    simclr_model = train_simclr(batch_size=256,
                                max_epochs=2000,
                                train_data=train_dataset_contrastive,
                                checkpoint_path=CHECKPOINT_PATH,
                                hidden_dim=128,
                                lr=5e-4,
                                temperature=0.07,
                                weight_decay=1e-4,
                                n_views=n_views)



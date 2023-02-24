import os
import matplotlib.pyplot as plt
import torchvision
from pytorch_lightning.strategies import DDPStrategy
from dotenv import load_dotenv
from OCT_dataset import OCTDataset, get_kaggle_imgs

from train import train_simclr, train_simclr_p
from transformation import ContrastiveTransformations, train_aug

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
    devices = torch.cuda.device_count()
    N_VIEWS = 2
    CV = 5,
    load_dotenv(dotenv_path="./data/.env")
    # Path to the folder where the datasets are
    DATASET_PATH = os.getenv('KAGGLE_FULL_DATASET_PATH')
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./kaggle_saved_models_full_8cores_acc_new_version/SimCLR/"
    # Path to style transferred images
    # NST_PATH = "data/nst_balanced.hdf5"
    # In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
    # workers as possible in a data loader, which corresponds to the number of CPU cores
    NUM_WORKERS = os.cpu_count() // 2

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)
    print("Number of workers:", NUM_WORKERS)
    classes = [("NORMAL", 0),
               ("AMD", 1),
               ("DME", 2)]
    train_dataset = OCTDataset(data_root=DATASET_PATH,
                               transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
                               classes=classes,
                               mode="train",
                               val_split=0.3,
                               # style_hdf5_path=NST_PATH,
                               dataset_func=get_kaggle_imgs,
                               )
    # print(set(np.array(range(1, PATIENTS + 1))) -set(choices))
    val_dataset = OCTDataset(data_root=DATASET_PATH,
                             transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
                             classes=classes,
                             mode="val",
                             val_split=0.3,
                             # style_hdf5_path=NST_PATH,
                             dataset_func=get_kaggle_imgs,
                             )
    print("len train:", len(train_dataset))
    print("len val: ", len(val_dataset))
    strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
    simclr_model = train_simclr(devices=devices,
                                strategy=strategy,
                                batch_size=min(len(train_dataset) // devices, 450),
                                # batch_size=4,
                                max_epochs=2000,
                                train_data=train_dataset,
                                checkpoint_path=CHECKPOINT_PATH,
                                hidden_dim=5 * 128,
                                feature_dim=128,
                                n_views=N_VIEWS,
                                gradient_accumulation_steps=1,
                                patience=25,
                                save_model_name="SimCLR",
                                monitor="train_loss",
                                # monitor="train_acc_mean_pos",
                                mode="min"
                                )

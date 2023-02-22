import os

import h5py
import torch
from dotenv import load_dotenv
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image
from OCT_dataset import OCTDataset, get_dfDict, ContrastiveTransformations, train_aug, get_kaggle_imgs

devices = torch.cuda.device_count()
N_VIEWS = 2
CV = 5
load_dotenv(dotenv_path="./data/.env")
# Path to the folder where the datasets are
DATASET_PATH = os.getenv('KAGGLE_BALANCED_DATASET_PATH')
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./kaggle_saved_models_very_balanced_2cores_acc/SimCLR/"
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
                           dataset_func=get_kaggle_imgs)
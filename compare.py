import math

import numpy as np
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import random_split
from torchvision.transforms import transforms, InterpolationMode
import os
import torch

from OCT_dataset import ContrastiveTransformations, OCTDataset, train_aug
from custom_dataset import CustomDataset
from simclr import SimCLR
from train import prepare_data_features, train_resnet, train_logreg


def _to_three_channel(x):
    return torch.cat([x, x, x], 0)


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    devices = torch.cuda.device_count()
    devices = 8
    N_VIEWS = 2
    CV = 5
    PATIENTS = 15
    cv_step = PATIENTS // CV
    # Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
    DATASET_PATH = "./2014_BOE_Srinivasan_2/Publication_Dataset"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./saved_models/"
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
    idx = np.array(range(1, cv_step + 1))

    for i in range(CV):
        train_dataset = OCTDataset(data_root="./2014_BOE_Srinivasan_2/Publication_Dataset/original data",
                                   img_suffix='.tif',
                                   transform=img_transforms,
                                   folders=idx,
                                   extra_folder_names="TIFFs/8bitTIFFs"
                                   )

        test_dataset = OCTDataset(data_root="./2014_BOE_Srinivasan_2/Publication_Dataset/original data",
                                  img_suffix='.tif',
                                  transform=img_transforms,
                                  folders=list(set(np.array(range(1, PATIENTS + 1))) - set(idx)),
                                  extra_folder_names="TIFFs/8bitTIFFs"
                                  )

        training_set, val_set = random_split(train_dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

        simclr_model = SimCLR.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR", "SimCLR_" + str(idx),
                                                                "SimCLR_" + str(idx) + ".ckpt"))
        batch_size = 64
        print("training data preparation")
        train_feats_simclr = prepare_data_features(model=simclr_model,
                                                   dataset=training_set,
                                                   device=device,
                                                   batch_size=batch_size)
        print("validation data preparation")
        val_feats_simclr = prepare_data_features(model=simclr_model,
                                                 dataset=val_set,
                                                 device=device,
                                                 batch_size=batch_size)
        print("testing data preparation")
        test_feats_simclr = prepare_data_features(model=simclr_model,
                                                  dataset=test_dataset,
                                                  device=device,
                                                  batch_size=batch_size)

        strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
        logreg_model, logreg_result = train_logreg(devices=devices,
                                                   strategy=strategy,
                                                   batch_size=batch_size,
                                                   train_feats_data=train_feats_simclr,
                                                   val_feats_data=val_feats_simclr,
                                                   test_feats_data=test_feats_simclr,
                                                   feature_dim=train_feats_simclr.tensors[0].shape[1],
                                                   num_classes=3,
                                                   checkpoint_path=CHECKPOINT_PATH + "LogisticRegression",
                                                   lr=1e-3,
                                                   weight_decay=1e-3,
                                                   max_epochs=100,
                                                   save_model_name="LogisticRegression" + str(idx))
        with open('accuracy_logreg.txt', 'a') as f:
            f.write("==================" + str(idx) + "==================")
            f.write('\n')
            f.write(str(logreg_result['train']))
            f.write('\n' + str(logreg_result['val']))
            f.write('\n' + str(logreg_result['test']))
            f.write('\n')

        strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
        resnet_model, resnet_result = train_resnet(devices=devices,
                                                   strategy=strategy,
                                                   batch_size=batch_size,
                                                   train_data=training_set,
                                                   val_data=val_set,
                                                   test_data=test_dataset,
                                                   lr=1e-3,
                                                   weight_decay=2e-4,
                                                   checkpoint_path=CHECKPOINT_PATH + "/ResNet",
                                                   max_epochs=100,
                                                   num_classes=3,
                                                   save_model_name="ResNet" + str(idx))

        with open('accuracy_resnet.txt', 'a') as f:
            f.write("==================" + str(idx) + "==================")
            f.write('\n')
            f.write(str(resnet_result['train']))
            f.write('\n' + str(resnet_result['val']))
            f.write('\n' + str(resnet_result['test']))
            f.write('\n')

        print(f"Accuracy on training set:{resnet_result['train']}")
        print(f"Accuracy on validation set: {resnet_result['val']}")
        print(f"Accuracy on test set: {resnet_result['test']}")

        idx += cv_step

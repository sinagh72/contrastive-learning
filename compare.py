import math

import numpy as np
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
        test_dataset = OCTDataset(data_root="./2014_BOE_Srinivasan_2/Publication_Dataset/original data",
                                  img_suffix='.tif',
                                  transform=img_transforms,
                                  folders=idx)

        train_dataset = OCTDataset(data_root="./2014_BOE_Srinivasan_2/Publication_Dataset/original data",
                                   img_suffix='.tif',
                                   transform=img_transforms,
                                   folders=list(set(np.array(range(1, PATIENTS + 1))) - set(idx)))

        simclr_model = SimCLR.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR", "SimCLR_" + str(idx),
                                                                "SimCLR_" + str(idx) + ".ckpt"))

        batch_size = 64
        train_feats_simclr = prepare_data_features(simclr_model, train_dataset, batch_size)
        test_feats_simclr = prepare_data_features(simclr_model, test_dataset, batch_size)
        log_every_n_steps = math.ceil(len(train_dataset) / batch_size)

        logreg_model, logreg_result = train_logreg(batch_size=batch_size,
                                                   train_feats_data=train_feats_simclr,
                                                   test_feats_data=test_feats_simclr,
                                                   feature_dim=train_feats_simclr.tensors[0].shape[1],
                                                   num_classes=3,
                                                   checkpoint_path=CHECKPOINT_PATH,
                                                   lr=1e-3,
                                                   weight_decay=1e-3,
                                                   max_epochs=100,
                                                   log_every_n_steps=log_every_n_steps,
                                                   save_model_name="LogisticRegression/LogisticRegression" + str(idx))

        print(f"Accuracy on training set: {100 * logreg_result['train']:4.2f}%")
        print(f"Accuracy on test set: {100 * logreg_result['test']:4.2f}%")

        resnet_model, resnet_result = train_resnet(batch_size=batch_size,
                                                   train_data=train_dataset,
                                                   test_data=test_dataset,
                                                   lr=1e-3,
                                                   weight_decay=2e-4,
                                                   checkpoint_path=CHECKPOINT_PATH,
                                                   max_epochs=100,
                                                   num_classes=3,
                                                   log_every_n_steps=log_every_n_steps,
                                                   save_model_name="ResNet/ResNet" + str(idx))

        print(f"Accuracy on training set: {100 * resnet_result['train']:4.2f}%")
        print(f"Accuracy on test set: {100 * resnet_result['test']:4.2f}%")

        idx += cv_step
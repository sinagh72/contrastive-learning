import math

from torchvision.transforms import transforms, InterpolationMode
import os
import torch

from custom_dataset import CustomDataset
from simclr import SimCLR
from train import prepare_data_features, train_resnet, train_logreg


def _to_three_channel(x):
    return torch.cat([x, x, x], 0)


if __name__ == "__main__":
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

    train_dataset = CustomDataset(data_root=DATASET_PATH + "/Train", img_suffix='.tif',
                                  transform=img_transforms)
    test_dataset = CustomDataset(data_root=DATASET_PATH + "/Test", img_suffix='.tif',
                                 transform=img_transforms)

    simclr_model = SimCLR.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR", "SimCLR.ckpt"))

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
                                               log_every_n_steps=log_every_n_steps)

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
                                               log_every_n_steps=log_every_n_steps)
    print(f"Accuracy on training set: {100 * resnet_result['train']:4.2f}%")
    print(f"Accuracy on test set: {100 * resnet_result['test']:4.2f}%")

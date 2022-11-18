from torchvision.transforms import transforms
import os
import torch

from custom_dataset import CustomDataset
from simclr import SimCLR
from train import prepare_data_features, train_resnet, train_logreg

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

    img_transforms = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,))])

    train_dataset_logreg = CustomDataset(data_root=DATASET_PATH + "/Train", img_suffix='.tif',
                                         transform=img_transforms)
    test_dataset_logreg = CustomDataset(data_root=DATASET_PATH + "/Test", img_suffix='.tif',
                                        transform=img_transforms)
    simclr_model = SimCLR.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR"))

    train_feats_simclr = prepare_data_features(simclr_model, train_dataset_logreg)
    test_feats_simclr = prepare_data_features(simclr_model, test_dataset_logreg)

    logreg_model, logreg_result = train_logreg(batch_size=64,
                                               train_feats_data=train_feats_simclr,
                                               test_feats_data=test_feats_simclr,
                                               feature_dim=train_feats_simclr.tensors[0].shape[1],
                                               num_classes=3,
                                               lr=1e-3,
                                               weight_decay=1e-3)

    print(f"Accuracy on training set: {100 * logreg_result['train']:4.2f}%")
    print(f"Accuracy on test set: {100 * logreg_result['test']:4.2f}%")

    resnet_model, resnet_result = train_resnet(batch_size=64,
                                               num_classes=3,
                                               lr=1e-3,
                                               weight_decay=2e-4,
                                               max_epochs=100)
    print(f"Accuracy on training set: {100 * resnet_result['train']:4.2f}%")
    print(f"Accuracy on test set: {100 * resnet_result['test']:4.2f}%")

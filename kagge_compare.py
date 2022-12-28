import numpy as np
from pytorch_lightning.strategies import DDPStrategy
from torchvision.transforms import transforms, InterpolationMode
import os
import torch

from OCT_dataset import OCTDataset, KaggleOCTDataset, ContrastiveTransformations, train_aug
from models.simclr import SimCLR
from train import prepare_data_features, train_resnet, train_logreg


def _to_three_channel(x):
    return torch.cat([x, x, x], 0)


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    devices = torch.cuda.device_count()
    devices = 8
    N_VIEWS = 2
    CV = 5
    # Path to the folder where the datasets are
    DATASET_PATH = "data/kaggle_dataset/train"
    # Path to the folder where the pretrained models are saved
    CHECKPOINT_PATH = "./kaggle_saved_models/"
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

    for i in range(CV):
        train_dataset = KaggleOCTDataset(data_root=DATASET_PATH,
                                         img_suffix='.jpeg',
                                         transform=img_transforms,
                                         classes=[("NORMAL", 0),
                                                  ("CNV", 1),
                                                  ("DME", 2),
                                                  ("DRUSEN", 3)],
                                         mode="train",
                                         cv=CV,
                                         cv_counter=i
                                         )
        val_dataset = KaggleOCTDataset(data_root=DATASET_PATH,
                                       img_suffix='.jpeg',
                                       transform=img_transforms,
                                       classes=[("NORMAL", 0),
                                                ("CNV", 1),
                                                ("DME", 2),
                                                ("DRUSEN", 3)],
                                       mode="val",
                                       cv=CV,
                                       cv_counter=i
                                       )

        test_dataset = KaggleOCTDataset(data_root=DATASET_PATH,
                                        img_suffix='.jpeg',
                                        transform=img_transforms,
                                        classes=[("NORMAL", 0),
                                                 ("CNV", 1),
                                                 ("DME", 2),
                                                 ("DRUSEN", 3)],
                                        mode="test",
                                        )

        simclr_model = SimCLR.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR", "SimCLR_" + str(i),
                                                                "SimCLR_" + str(i) + ".ckpt"))
        batch_size = 64
        print("training data preparation")
        train_feats_simclr = prepare_data_features(model=simclr_model,
                                                   dataset=train_dataset,
                                                   device=device,
                                                   batch_size=batch_size)
        print("validation data preparation")
        val_feats_simclr = prepare_data_features(model=simclr_model,
                                                 dataset=val_dataset,
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
                                                   num_classes=4,
                                                   checkpoint_path=CHECKPOINT_PATH + "LogisticRegression",
                                                   lr=1e-3,
                                                   weight_decay=1e-3,
                                                   max_epochs=100,
                                                   save_model_name="LogisticRegression" + str(i))
        with open('log/kaggle_accuracy_logreg.txt', 'a') as f:
            f.write("==================" + str(i) + "==================")
            f.write('\n')
            f.write(str(logreg_result['train']))
            f.write('\n' + str(logreg_result['val']))
            f.write('\n' + str(logreg_result['test']))
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
                                                   num_classes=4,
                                                   save_model_name="ResNet" + str(i))

        with open('log/kaggle_accuracy_resnet.txt', 'a') as f:
            f.write("==================" + str(i) + "==================")
            f.write('\n')
            f.write(str(resnet_result['train']))
            f.write('\n' + str(resnet_result['val']))
            f.write('\n' + str(resnet_result['test']))
            f.write('\n')

        print(f"Accuracy on training set:{resnet_result['train']}")
        print(f"Accuracy on validation set: {resnet_result['val']}")
        print(f"Accuracy on test set: {resnet_result['test']}")


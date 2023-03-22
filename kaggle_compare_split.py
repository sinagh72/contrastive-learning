import os

import torch
from dotenv import load_dotenv
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import random_split
from torchvision.transforms import transforms, InterpolationMode

from OCT_dataset import OCTDataset, get_kaggle_imgs, get_duke_imgs
from train import train_resnet, train_simclr_p

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    devices = torch.cuda.device_count()
    N_VIEWS = 2
    load_dotenv(dotenv_path="./data/.env")
    # Path to the folder where the datasets are
    DATASET_PATH = os.getenv('KAGGLE_BALANCED_DATASET_PATH')
    # Path to load simclr and to save resnet and linear models
    CHECKPOINT_PATH = "trained_models/kaggle_balanced_portion_top5/"
    # Path to style transferred image
    # NST_PATH = "data/nst_balanced.hdf5"

    TEST_DATASET_PATH = os.getenv("KAGGLE_COMPARE_TEST_DATASET_PATH")
    # In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
    # workers as possible in a data loader, which corresponds to the number of CPU cores
    NUM_WORKERS = os.cpu_count()

    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Number of workers:", NUM_WORKERS)

    img_transforms = transforms.Compose([transforms.Resize(size=(128, 128), interpolation=InterpolationMode.BICUBIC),
                                         transforms.Grayscale(3),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5,)),
                                         ])

    classes = [("NORMAL", 0),
               ("AMD", 1),
               ("DME", 2)]

    test_dataset = OCTDataset(data_root=TEST_DATASET_PATH,
                              transform=img_transforms,
                              dataset_func=get_duke_imgs,
                              classes=classes,
                              ignore_folders=[],
                              sub_folders_name="TIFFs/8bitTIFFs",

                              )

    batch_size = 64
    mode = "max"
    monitor = "val_loss"
    patience = 5
    max_epochs = 100

    i = 0.1
    while i <= 1:
        train_val_dataset = OCTDataset(data_root=DATASET_PATH,
                                       transform=img_transforms,
                                       dataset_func=get_kaggle_imgs,
                                       classes=classes,
                                       mode="train",
                                       val_split=round(1 - i, 1),
                                       # style_hdf5_path=NST_PATH,
                                       )
        print(len(train_val_dataset))
        train_dataset, val_dataset = train_val_dataset.split(0.1)
        print("==================Resnet==================")
        log_name_suffix = "kaggle_portion_top5"
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
                                                   max_epochs=max_epochs,
                                                   classes=classes,
                                                   save_model_name="ResNet_" + str(round(i, 1)),
                                                   mode=mode,
                                                   monitor=monitor,
                                                   patience=patience)

        file_mode = "a" if os.path.exists(f'log/{log_name_suffix}_resnet_{batch_size}_{str(round(i, 1))}.txt') else "w"
        with open(f'log/{log_name_suffix}_resnet_{batch_size}_{str(round(i, 1))}.txt', file_mode) as f:
            f.write("====================================")
            f.write('\n')
            f.write(str(resnet_result['train']))
            f.write('\n' + str(resnet_result['val']))
            f.write('\n' + str(resnet_result['test']))
            f.write('\n')
        del resnet_model
        torch.cuda.empty_cache()
        # trained_dataset = OCTDataset(data_root=DATASET_PATH,
        #                              transform=img_transforms,
        #                              classes=classes,
        #                              mode="train",
        #                              val_split=0.3,
        #                              # style_hdf5_path=NST_PATH,
        #                              dataset_func=get_kaggle_imgs,
        #                              )

        # train_dataset, val_dataset = random_split(train_val_dataset, [0.9, 0.1],
        #                                           generator=torch.Generator().manual_seed(42))

        print(f"training data len: {len(train_dataset)}")
        print(f"validation data len: {len(val_dataset)}")
        print(f"testing data len: {len(test_dataset)}")

        print("==================SimCLR Model==================")
        strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
        simclrp_model, simclrp_result = train_simclr_p(devices=devices,
                                                       strategy=strategy,
                                                       batch_size=batch_size,
                                                       train_dataset=train_dataset,
                                                       val_dataset=val_dataset,
                                                       test_dataset=test_dataset,
                                                       classes=classes,
                                                       checkpoint_path=CHECKPOINT_PATH,
                                                       lr=1e-3,
                                                       feature_dim=128,
                                                       weight_decay=1e-3,
                                                       max_epochs=max_epochs,
                                                       mode=mode,
                                                       monitor=monitor,
                                                       patience=patience,
                                                       freeze_p=0.0,
                                                       encoder_path="SimCLR_" + str(1.0),
                                                       save_model_name="SimCLR_portion_" + str(round(i, 1)),
                                                       )

        file_mode = "a" if os.path.exists(
            f'log/{log_name_suffix}_simclrp_{batch_size}_{str(round(i, 1))}.txt') else "w"
        with open(f'log/{log_name_suffix}_simclrp_{batch_size}_{str(round(i, 1))}.txt', file_mode) as f:
            f.write("====================================")
            f.write('\n')
            f.write(str(simclrp_result['train']))
            f.write('\n' + str(simclrp_result['val']))
            f.write('\n' + str(simclrp_result['test']))
            f.write('\n')

        i += 0.1
        i = round(i, 1)
        del simclrp_model
        torch.cuda.empty_cache()
# print(f"{metric} on training set:{resnet_result['train']}")
# print(f"{metric} on validation set: {resnet_result['val']}")
# print(f"{metric} on test set: {resnet_result['test']}")

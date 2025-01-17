import os

import torch
from dotenv import load_dotenv
from pytorch_lightning.strategies import DDPStrategy
from torchvision.transforms import transforms, InterpolationMode

from OCT_dataset import OCTDataset, get_kaggle_imgs, get_duke_imgs
from models.simclr import SimCLR
from train import prepare_data_features, train_resnet, train_linear_model

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    devices = torch.cuda.device_count()
    N_VIEWS = 2
    CV = 5
    load_dotenv(dotenv_path="./data/.env")
    # Path to the folder where the datasets are
    DATASET_PATH = os.getenv('KAGGLE_BALANCED_DATASET_PATH')
    # Path to load simclr and to save resnet and linear models
    CHECKPOINT_PATH = "trained_models/kaggle_saved_models_very_balanced_2cores_acc/"
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

    metric = "accuracy"
    log_name_suffix = "kaggle_very_balanced_2cores_acc"
    batch_size = 128

    for i in range(CV):
        train_dataset = OCTDataset(data_root=DATASET_PATH,
                                   transform=img_transforms,
                                   classes=classes,
                                   mode="train",
                                   # nst_path=NST_PATH,
                                   dataset_func=get_kaggle_imgs,
                                   cv=CV,
                                   cv_counter=i
                                   )
        val_dataset = OCTDataset(data_root=DATASET_PATH,
                                 transform=img_transforms,
                                 dataset_func=get_kaggle_imgs,
                                 classes=classes,
                                 mode="val",
                                 # nst_path=NST_PATH,
                                 cv=CV,
                                 cv_counter=i
                                 )

        test_dataset = OCTDataset(data_root=TEST_DATASET_PATH,
                                  transform=img_transforms,
                                  dataset_func=get_duke_imgs,
                                  classes=classes,
                                  ignore_folders=[],
                                  sub_folders_name="TIFFs/8bitTIFFs",
                                  
                                  )

        simclr_model = SimCLR.load_from_checkpoint(os.path.join(CHECKPOINT_PATH, "SimCLR", "SimCLR_" + str(i),
                                                                "SimCLR_" + str(i) + ".ckpt"))
        print("training data preparation")
        print(f"training data len: {len(train_dataset)}")
        print(f"validation data len: {len(val_dataset)}")
        print(f"testing data len: {len(test_dataset)}")
        train_feats_simclr = prepare_data_features(model=simclr_model,
                                                   dataset=train_dataset,
                                                   device=device,
                                                   batch_size=batch_size,
                                                   num_workers=4)
        print("validation data preparation")
        val_feats_simclr = prepare_data_features(model=simclr_model,
                                                 dataset=val_dataset,
                                                 device=device,
                                                 batch_size=batch_size,
                                                 num_workers=4)
        print("testing data preparation")
        test_feats_simclr = prepare_data_features(model=simclr_model,
                                                  dataset=test_dataset,
                                                  device=device,
                                                  batch_size=batch_size,
                                                  num_workers=4)

        print("==================Resnet==================")

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
                                                   classes=classes,
                                                   # metric=metric,
                                                   save_model_name="ResNet" + str(i))

        file_mode = "a" if os.path.exists(f'log/{log_name_suffix}_{metric}_resnet_{batch_size}.txt') else "w"
        with open(f'log/{log_name_suffix}_{metric}_resnet_{batch_size}.txt', file_mode) as f:
            f.write("==================" + str(i) + "==================")
            f.write('\n')
            f.write(str(resnet_result['train']))
            f.write('\n' + str(resnet_result['val']))
            f.write('\n' + str(resnet_result['test']))
            f.write('\n')

            print("==================Linear Model==================")
            strategy = None if devices == 1 else DDPStrategy(find_unused_parameters=False)
            lmodel_model, lmodel_result = train_linear_model(devices=devices,
                                                             strategy=strategy,
                                                             batch_size=batch_size,
                                                             train_feats_data=train_feats_simclr,
                                                             val_feats_data=val_feats_simclr,
                                                             test_feats_data=test_feats_simclr,
                                                             feature_dim=train_feats_simclr.tensors[0].shape[1],
                                                             classes=classes,
                                                             checkpoint_path=CHECKPOINT_PATH + "LinearModel",
                                                             lr=1e-3,
                                                             weight_decay=1e-3,
                                                             max_epochs=100,
                                                             # metric=metric,
                                                             save_model_name="LinearModel" + str(i))

            file_mode = "a" if os.path.exists(f'log/{log_name_suffix}_{metric}_lmodel_{batch_size}.txt') else "w"
            with open(f'log/{log_name_suffix}_{metric}_lmodel_{batch_size}.txt', file_mode) as f:
                f.write("==================" + str(i) + "==================")
                f.write('\n')
                f.write(str(lmodel_result['train']))
                f.write('\n' + str(lmodel_result['val']))
                f.write('\n' + str(lmodel_result['test']))
                f.write('\n')

        #
        # print(f"{metric} on training set:{resnet_result['train']}")
        # print(f"{metric} on validation set: {resnet_result['val']}")
        # print(f"{metric} on test set: {resnet_result['test']}")

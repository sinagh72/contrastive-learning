from torchvision.transforms import transforms, InterpolationMode

from OCT_dataset import OCTDataset
from kaggle_compare import _to_three_channel

TEST_DATASET_PATH = "data/2014_BOE_Srinivasan_2/Publication_Dataset/original_data"
img_transforms = transforms.Compose([transforms.Resize(size=(128, 128), interpolation=InterpolationMode.BICUBIC),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                     transforms.Lambda(_to_three_channel)])

test_dataset = OCTDataset(data_root=TEST_DATASET_PATH,
                          transform=img_transforms,
                          discard_folders=[],
                          extra_folder_names="TIFFs/8bitTIFFs"
                          )

print(len(test_dataset))
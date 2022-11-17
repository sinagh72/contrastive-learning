from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode
import torch
import os
import numpy as np
from PIL import Image

IMAGE_SIZE = (512, 496)


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


def train_aug(image):
    transfrom = transforms.Compose([transforms.Resize(128, InterpolationMode.BICUBIC),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomResizedCrop(size=128, scale=(0.25, 0.75),
                                                                 interpolation=InterpolationMode.BICUBIC),
                                    transforms.RandomApply([
                                        transforms.ColorJitter(brightness=0.5,
                                                               contrast=0.5,
                                                               saturation=0.5,
                                                               hue=0.1)
                                    ], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.GaussianBlur(kernel_size=9),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                    ])
    img = transfrom(image.copy())
    return img


class CustomDataset(Dataset):

    def __init__(self, data_root, mode, img_format="L", img_suffix='.png', transform=train_aug, img_size=IMAGE_SIZE):
        self.data_root = data_root
        self.img_suffix = img_suffix
        self.transform = transform
        self.mode = mode
        self.img_size = img_size
        self.img_format = img_format
        self.img_ids = self.get_img_ids(self.data_root)

    def __getitem__(self, index):
        img = self.load_img(index)
        img = self.transform(img)
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id = self.img_ids[index]
        """
        if patient has:
         - AMD    -> 1
         - DEM    -> 2
         - Normal -> 0
        """
        ann = 1 if "AMD" in img_id else 2 if "DEM" in img_id else 0

        results = dict(img_id=img_id, img=img, y_true=ann)

        return results

    def __len__(self):
        """
        The __len__ should be overridden when the parent is Dataset to return the number of elements of the dataset

        Return:
            - returns the size of the dataset
        """
        return len(self.img_ids)

    def get_img_ids(self, data_root: str):
        img_filename_list = os.listdir(data_root)
        img_ids = []
        for img_file in img_filename_list:
            root = os.path.join(data_root, img_file, "TIFFs", "8bitTIFFs")
            img_ids += [os.path.join(root, str(id.split('.')[0])) for id in os.listdir(root)]
        return img_ids

    def load_img(self, index):
        img_id = self.img_ids[index]
        img = Image.open(img_id + self.img_suffix).convert(self.img_format)
        return img

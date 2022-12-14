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
        return [self.base_transforms(x) for _ in range(self.n_views)]


def train_aug(image):
    transfrom = transforms.Compose([transforms.Resize(128, InterpolationMode.BICUBIC),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(degrees=45),
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


class OCTDataset(Dataset):

    def __init__(self, data_root, img_type="L", img_suffix='.png', transform=train_aug, img_size=IMAGE_SIZE,
                 folders=None, mode="train", extra_folder_names="", classes=None):
        if classes is None:
            classes = [("NORMAL", 0),
                       ("AMD", 1),
                       ("DME", 2)]
        self.data_root = data_root
        self.extra_folder_names = extra_folder_names
        self.img_suffix = img_suffix
        self.transform = transform
        self.img_size = img_size
        self.img_type = img_type
        self.folders = folders
        self.mode = mode
        self.img_ids = self.get_img_ids(self.data_root)
        self.classes = classes

    def __getitem__(self, index):
        img = self.load_img(index)
        img = self.transform(img)
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        img_id = self.img_ids[index]
        """
        if patient has:
         - AMD    -> 1
         - DME    -> 2
         - Normal -> 0
        """
        ann = 0
        for c, v in self.classes:
            if c in img_id:
                ann = v
                break

        img_id = img_id.replace("\\", "/")

        results = dict(img_id=img_id, img_folder=img_id.split(self.data_root)[1].split("/")[1], img=img, y_true=ann)

        return results

    def __len__(self):
        """
        The __len__ should be overridden when the parent is Dataset to return the number of elements of the dataset

        Return:
            - returns the size of the dataset
        """
        return len(self.img_ids)

    def get_img_ids(self, data_root: str):
        img_filename_list = os.listdir(os.path.join(data_root))
        img_ids = []
        for img_file in img_filename_list:
            if any(item == int(img_file.replace("AMD", "").replace("NORMAL", "").replace("DME", ""))
                   for item in self.folders):
                continue
            folder = os.path.join(data_root, img_file, self.extra_folder_names)
            img_ids += [os.path.join(folder, id) for id in os.listdir(folder)]
            # if "AMD" in folder:
            #     counts = len(os.listdir(os.path.join(data_root, "AMD")))
            #     for img in new_data:
            #         Image.open(img + ".tif").save(os.path.join(data_root, "AMD", "AMD_" + str(counts) + ".tif"))
            #         counts += 1
            # elif "DME" in root:
            #     counts = len(os.listdir(os.path.join(data_root, "DME")))
            #     for img in new_data:
            #         Image.open(img + ".tif").save(os.path.join(data_root, "DME", "DME_" + str(counts) + ".tif"))
            #         counts += 1
            #
            # elif "NORMAL" in root:
            #     counts = len(os.listdir(os.path.join(data_root, "Normal")))
            #     for img in new_data:
            #         Image.open(img + ".tif").save(os.path.join(data_root, "NORMAL", "NORMAL_" + str(counts) + ".tif"))
            #         counts += 1
        return img_ids

    def load_img(self, index):
        img_id = self.img_ids[index]
        img = Image.open(img_id).convert(self.img_type)
        return img

import random
import warnings

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import transforms, InterpolationMode
import torch
import os
from PIL import Image
from natsort import natsorted

IMAGE_SIZE = (512, 496)


class ContrastiveTransformations(object):

    def __init__(self, transformations, n_views=2):
        self.transformations = transformations
        self.n_views = n_views

    def __call__(self, x, apply_views: bool = True):
        if apply_views:
            return [self.transformations(x) for _ in range(self.n_views)]
        return self.transformations(x)


def train_aug(img):
    transform = transforms.Compose([transforms.Resize((128, 128), InterpolationMode.BICUBIC),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=45),
                                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                    # transforms.RandomResizedCrop(size=128, scale=(0.25, 0.75),
                                    #                              interpolation=InterpolationMode.BICUBIC),
                                    # transforms.RandomApply([
                                    #     transforms.ColorJitter(brightness=0.5,
                                    #                            contrast=0.5,
                                    #                            saturation=0.5,
                                    #                            hue=0.1)
                                    # ], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.GaussianBlur(kernel_size=9),
                                    transforms.Grayscale(3),
                                    
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                    ])
    img = transform(img.copy())
    return img


class OCTDataset(Dataset):

    def __init__(self, data_root, img_type="L", transform=None, img_size=IMAGE_SIZE, classes=None,
                 dataset_func=None, style_hdf5_path=None, nst_prob=1, **kwargs):
        if classes is None:
            classes = [("NORMAL", 0),
                       ("AMD", 1),
                       ("DME", 2)]
        self.data_root = data_root
        self.transform = transform
        self.img_size = img_size
        self.img_type = img_type
        self.classes = classes
        self.style_hdf5_path = style_hdf5_path
        self.img_paths = dataset_func(self.data_root, **kwargs)
        self.nst_prob = nst_prob
        if self.style_hdf5_path is not None:
            self.nst_img, self.nst_img_dic, self.nst_img_names = get_dfDict(style_hdf5_path)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img_path = img_path.replace("\\", "/")

        if random.uniform(0, 1) < self.nst_prob and self.style_hdf5_path is not None:
            img = self.load_nst_img(img_path)
        else:
            image = self.load_img(img_path)
            if self.transform:
                img = self.transform(image)
            else:
                img = [torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float()]
                # print(img.shape)
            # image.show()
        # img = torch.from_numpy(img).permute(2, 0, 1).float()
        """
        if patient has:
         - AMD    -> 1
         - DME    -> 2
         - Normal -> 0
        """
        img_path = self.img_paths[index]
        label = 0
        for c, v in self.classes:
            if c in img_path:
                label = v
                break

        # {'img_path': full path to the img, 'img_folder': folder name, 'img_name': img name, 'img': PIL.Image,
        # 'label': depends on the classes }
        results = dict(img_path=img_path, img_folder=img_path.split(self.data_root)[1].split("/")[0],
                       img_name=img_path.split(self.data_root)[1].split("/")[1], img=img, label=label)
        return results

    def __len__(self):
        """
        The __len__ should be overridden when the parent is Dataset to return the number of elements of the dataset

        Return:
            - returns the size of the dataset
        """
        return len(self.img_paths)

    def load_img(self, img_path):
        img = Image.open(img_path).convert(self.img_type)
        return img

    def load_nst_img(self, img_path):
        # iteratively transform all images that are generated using NST
        selected_idx = np.random.choice(self.nst_img_dic[img_path].index.values, self.transform.n_views, replace=False)
        imgs = [self.nst_img[i] for i in selected_idx]
        for i in imgs:
            Image.fromarray(i).show()
        images = [self.transform(Image.fromarray(i), False) for i in imgs]
        # select n_views of them
        # Note: If n_views > len(generated nst for each image) then the image will be replicated!
        return images


def get_duke_imgs(data_root: str, **kwargs):
    """

    :param data_root:
    :param kwargs:
        - ignore_folders (np.array): indices of files to ignore
        - sub_folders_name (str): path containing the subfolders
    :return:
    """
    img_filename_list = os.listdir(os.path.join(data_root))
    imgs_path = []
    for img_file in img_filename_list:
        if any(item == int(img_file.replace("AMD", "").replace("NORMAL", "").replace("DME", ""))
               for item in kwargs["ignore_folders"]):
            continue
        folder = os.path.join(data_root, img_file, kwargs["sub_folders_name"])
        imgs_path += [os.path.join(folder, id) for id in os.listdir(folder)]
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
    return imgs_path


def get_kaggle_imgs(data_root: str, **kwargs):
    img_filename_list = os.listdir(os.path.join(data_root))
    imgs_path = []
    for img_file in img_filename_list:
        img_file_path = os.path.join(data_root, img_file)
        # sort lexicographically the img names in the img_file directory
        img_names = natsorted(os.listdir(img_file_path))
        # split the first part of image
        imgs_dict = {}
        for img in img_names:
            img_count = img.split("-")[2]
            img_name = img.replace(img_count, "")
            if img_name in imgs_dict:
                imgs_dict[img_name] += [img_count]
            else:
                imgs_dict[img_name] = [img_count]
        if kwargs["mode"] == "test":
            for key, val in imgs_dict.items():
                imgs_path += [img_file_path + "/" + key + count for count in val]
        else:
            cv_len = len(imgs_dict.keys()) // kwargs["cv"]
            start_idx = kwargs["cv_counter"] * cv_len
            end_idx = start_idx + cv_len if kwargs["cv_counter"] < kwargs["cv"] - 1 else len(imgs_dict.keys())
            if kwargs["cv"] == 1:
                # TODO: add the split percentage argument
                warnings.warn("Cross validation is 1! valid split percentage not implemented yet!")
                pass
            keys = list(imgs_dict.keys())
            for key, val in imgs_dict.items():
                if key not in keys[start_idx:end_idx] and kwargs["mode"] == "train":
                    imgs_path += [img_file_path + "/" + key + count for count in val]
                elif key in keys[start_idx:end_idx] and kwargs["mode"] == "val":
                    imgs_path += [img_file_path + "/" + key + count for count in val]
    return imgs_path


def get_dr_imgs(data_root: str, **kwargs):
    img_folders = os.listdir(os.path.join(data_root))
    imgs_dict = {}
    img_ids = []
    for img_folder in img_folders:
        imgs = natsorted(os.listdir(os.path.join(data_root, img_folder)))
        for img in imgs:
            img_path = os.path.join(data_root, img_folder, img)
            patient_name = img.split("_")[1] + img.split("_")[2]
            if patient_name in imgs_dict:
                imgs_dict[patient_name] += [img_path]
            else:
                imgs_dict[patient_name] = [img_path]
        if kwargs["mode"] == "test":
            for key, val in imgs_dict.items():
                img_ids += [img_p for img_p in val]
        else:
            cv_len = len(imgs_dict.keys()) // kwargs["cv"]
            start_idx = kwargs["cv_counter"] * cv_len
            end_idx = start_idx + cv_len if kwargs["cv_counter"] < kwargs["cv"] - 1 else len(imgs_dict.keys())
            keys = list(imgs_dict.keys())
            for key, val in imgs_dict.items():
                if key not in keys[start_idx:end_idx] and kwargs["mode"] == "train":
                    img_ids += [img_p for img_p in val]
                elif key in keys[start_idx:end_idx] and kwargs["mode"] == "val":
                    img_ids += [img_p for img_p in val]
    return img_ids


def splitting(img_name: str, token: str):
    return img_name.split(token)[1].replace("-", "", 1)


def get_dfDict(path2hdf5):
    h5 = h5py.File(path2hdf5)
    names = [i.decode('UTF-8') for i in h5['names']]
    paths = [i.decode('UTF-8') for i in h5['paths']]
    imgs = [i for i in h5['img']]

    df = pd.DataFrame(columns=['names', 'paths'])
    df.names = names
    df.paths = paths

    df_dict = {}
    for name, path in df.groupby('paths'):
        df_dict[name] = path

    return imgs, df_dict, names

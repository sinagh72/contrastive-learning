import math
import random
import warnings
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
from natsort import natsorted
import subsetsum as sb

IMAGE_SIZE = (512, 496)


def split_data(img_paths, val_split, mode, classes):
    img_paths_out = []
    for c in classes:
        filtered = list(filter(lambda k: c[0] in k, img_paths))
        img_dict = {}
        for img in filtered:
            img_count = img.split("-")[2]
            img_name = img.replace(img_count, "")
            if img_name in img_dict:
                img_dict[img_name].append(img_count)
            else:
                img_dict[img_name] = [img_count]

        num_visits = [len(n) for n in img_dict.values()]
        total_imgs = sum(num_visits)
        val_num = math.floor(total_imgs * val_split)
        for solution in sb.solutions(num_visits, val_num):
            # `solution` contains indices of elements in `nums`
            subset = [i for i in solution]
            break
        keys = list(img_dict.keys())
        counter = -1
        for idx in subset:
            if mode == "val":
                img_paths_out += [keys[idx] + count for count in img_dict[keys[idx]]]

        for key, val in img_dict.items():
            counter += 1
            if counter in subset:
                continue
            if mode == "train":
                img_paths_out += [key + count for count in val]
        print(mode, c[0], len(img_paths_out))
    return img_paths_out


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
        if "img_paths" in kwargs:
            self.img_paths = split_data(kwargs["img_paths"], val_split=kwargs["val_split"], mode=kwargs["mode"],
                                        classes=self.classes)
        else:
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
            if self.transform is not None:
                img = self.transform(image)
            else:
                # throws error
                img = image
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
        # img_folder = img_path.split(self.data_root)[1].split("/")[0],

        results = dict(img_path=img_path, img_name=img_path.split(self.data_root)[1].split("/")[1], img=img,
                       label=label)
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

    def split(self, val_split=0.1):
        train_dataset = OCTDataset(data_root=self.data_root, img_type=self.img_type, transform=self.transform,
                                   img_size=IMAGE_SIZE, classes=self.classes,
                                   style_hdf5_path=self.style_hdf5_path, mode="train", val_split=val_split,
                                   img_paths=self.img_paths)

        val_dataset = OCTDataset(data_root=self.data_root, img_type=self.img_type, transform=self.transform,
                                 img_size=IMAGE_SIZE, classes=self.classes,
                                 style_hdf5_path=self.style_hdf5_path, mode="val", val_split=val_split,
                                 img_paths=self.img_paths)

        return train_dataset, val_dataset


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
    img_paths = []
    for img_file in img_filename_list:
        img_file_path = os.path.join(data_root, img_file)
        # sort lexicographically the img names in the img_file directory
        img_names = natsorted(os.listdir(img_file_path))
        # split the first part of image
        # dictionary{patient_id:[list of visits]}
        img_dict = {}
        for img in img_names:
            img_count = img.split("-")[2]
            img_name = img.replace(img_count, "")
            if img_name in img_dict:
                img_dict[img_name] += [img_count]
            else:
                img_dict[img_name] = [img_count]
        if kwargs["mode"] == "test":
            for key, val in img_dict.items():
                img_paths += [img_file_path + "/" + key + count for count in val]
        elif "cv" in kwargs:
            cv_len = len(img_dict.keys()) // kwargs["cv"]
            start_idx = kwargs["cv_counter"] * cv_len
            end_idx = start_idx + cv_len if kwargs["cv_counter"] < kwargs["cv"] - 1 else len(img_dict.keys())
            if kwargs["cv"] == 1:
                # TODO: add the split percentage argument
                warnings.warn("Cross validation is 1! valid split percentage not implemented yet!")
                pass
            keys = list(img_dict.keys())
            for key, val in img_dict.items():
                if key not in keys[start_idx:end_idx] and kwargs["mode"] == "train":
                    img_paths += [img_file_path + "/" + key + count for count in val]
                elif key in keys[start_idx:end_idx] and kwargs["mode"] == "val":
                    img_paths += [img_file_path + "/" + key + count for count in val]
        elif "val_split" in kwargs:
            # sort the patients based on their number of visits
            # kwargs[split] is a value between (0,1)
            # find the len of dataset
            num_visits = [len(n) for n in img_dict.values()]
            total_imgs = sum(num_visits)
            val_num = math.floor(total_imgs * kwargs["val_split"])
            if val_num != 0:
                for solution in sb.solutions(num_visits, val_num):
                    # `solution` contains indices of elements in `nums`
                    subset = [i for i in solution]
                    break
                keys = list(img_dict.keys())
                for idx in subset:
                    if kwargs["mode"] == "val":
                        img_paths += [img_file_path + "/" + keys[idx] + count for count in img_dict[keys[idx]]]
            else:
                subset = []
            counter = -1
            for key, val in img_dict.items():
                counter += 1
                if counter in subset:
                    continue
                if kwargs["mode"] == "train":
                    img_paths += [img_file_path + "/" + key + count for count in val]
        print(f"mode: {kwargs['mode']}:{img_file_path}", len(img_paths))
    return img_paths


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

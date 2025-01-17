"""
Find the difference between two folders
To check whether the segmentation (classification) and image folders have the same images
"""
import os

import cv2
import numpy as np


def diff(first_dir, second_dir, first_dir_img_format=".png", second_dir_img_format=".png", remove_diff=False):
    """
    Finds the images that common in both first_dir and second_dir based on their names disregarding their format.
     It is possible to remove those images that are ont inside their intersection.

    Args:
        -first_dir (str): path to the first directory
        - first_dir_img_format (str): the format of the images inside the first_dir
        -second_dir (str): path to the second directory
        - second_dir_img_format (str): the format of the images inside the second_dir
        -remove_diff (bool): whether to remove images that are not in their intersection
    """
    d1 = sorted(os.listdir(first_dir))
    d2 = sorted(os.listdir(second_dir))
    f1 = []
    for f in d1:
        f1.append(f.split(first_dir_img_format)[0])

    f2 = []
    for f in d2:
        f2.append(f.split(second_dir_img_format)[0])

    intersection = list(set(f2).intersection(set(f1)))
    print("intersection between two files", len(intersection))

    if remove_diff:
        for f in intersection:
            os.remove(os.path.join(second_dir, f) + second_dir_img_format)

        # for f in intersection:
        #     os.remove(os.path.join(first_dir, f) + first_dir_img_format)


def mse(img1, img2):
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if h1 != h2 or w1 != w2:
        return 100
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff ** 2)
    mse = err / (float(h1 * w1))
    return mse


def check_duplicate_all(train_dir):
    folders = os.listdir(train_dir)
    imgs = []
    for f in folders:
        imgs += os.listdir(os.path.join(train_dir, f))
    for i in range(0, len(imgs) - 1):
        img1 = cv2.imread(os.path.join(train_dir, imgs[i].split("-")[0], imgs[i]), cv2.IMREAD_GRAYSCALE)
        for j in range(i + 1, len(imgs)):
            img2 = cv2.imread(os.path.join(train_dir, imgs[j].split("-")[0], imgs[j]), cv2.IMREAD_GRAYSCALE)
            if mse(img1, img2) == 0:
                print(imgs[i], imgs[j])


if __name__ == "__main__":
    # diff(first_dir="C:/Phd/projects/contrastive learning/data/kaggle_dataset/test/CNV",
    #      second_dir="C:/Phd/projects/contrastive learning/data/kaggle_dataset/train/CNV",
    #      first_dir_img_format=".jpeg",
    #      second_dir_img_format=".jpeg",
    #      remove_diff=True)
    check_duplicate_all(train_dir="./data/kaggle/kaggle_dataset_full")

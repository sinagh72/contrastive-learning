import numbers
import os
import time
from numpy import random

import cv2
import numpy as np
from PIL import ImageOps
from dotenv import load_dotenv
from torchvision.transforms import transforms

from OCT_dataset import OCTDataset, get_duke_imgs, get_kaggle_imgs
from transformation import train_transformation3




load_dotenv(dotenv_path="./data/.env")
DATASET_PATH = os.getenv('KAGGLE_FULL_DATASET_PATH')
folders = os.listdir(DATASET_PATH)
classes = [("NORMAL", 0),
           ("AMD", 1),
           ("DME", 2)]

trans = train_transformation3()
train_dataset = OCTDataset(data_root=DATASET_PATH,
                           transform=trans,
                           classes=classes,
                           mode="train",
                           val_split=0,
                           # nst_path=NST_PATH,
                           dataset_func=get_kaggle_imgs,
                           )
topil = transforms.ToPILImage()
trans(train_dataset[1]["img"]).show()

for i in train_dataset:
    out = trans(i["img"])
    cv2.imshow("test", np.asarray(out))
    cv2.waitKey(0)
    # topil(out).show()
# AMD = 0
# DME = 0
# NORMAL = 0
#
# AMD_t = 0
# DME_t = 0
# NORMAL_t = 0
# for sub_folder in folders:
#     if "AMD" in sub_folder and sub_folder != "AMD15" and sub_folder != "AMD14" and sub_folder != "AMD13":
#         AMD += len([i for i in os.listdir(os.path.join(DATASET_PATH, sub_folder, "TIFFs", "8bitTIFFs"))])
#     elif "DME" in sub_folder and sub_folder != "DME15" and sub_folder != "DME14" and sub_folder != "DME13":
#         DME += len([i for i in os.listdir(os.path.join(DATASET_PATH, sub_folder, "TIFFs", "8bitTIFFs"))])
#     elif "NORMAL" in sub_folder and sub_folder != "NORMAL13" and sub_folder != "NORMAL14" and sub_folder != "NORMAL15":
#         NORMAL += len([i for i in os.listdir(os.path.join(DATASET_PATH, sub_folder, "TIFFs", "8bitTIFFs"))])
#     elif sub_folder == "AMD15" or sub_folder == "AMD14" or sub_folder == "AMD13":
#         AMD_t += len([i for i in os.listdir(os.path.join(DATASET_PATH, sub_folder, "TIFFs", "8bitTIFFs"))])
#     elif sub_folder == "DME13" or sub_folder == "DME14" or sub_folder == "DME15":
#         DME_t += len([i for i in os.listdir(os.path.join(DATASET_PATH, sub_folder, "TIFFs", "8bitTIFFs"))])
#     elif sub_folder == "NORMAL13" or sub_folder == "NORMAL14" or sub_folder == "NORMAL15":
#         NORMAL_t += len([i for i in os.listdir(os.path.join(DATASET_PATH, sub_folder, "TIFFs", "8bitTIFFs"))])
# print(AMD_t+AMD, DME_t+DME, NORMAL_t+NORMAL)
# print(AMD, DME, NORMAL)
# print(AMD_t, DME_t, NORMAL_t)

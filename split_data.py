import math
import os
import mmcv
import random
from PIL import Image


def move_data(input_dict: dict):

    for f in input_dict["files"]:
        img_name = input_dict["img_name"] + input_dict["image_format"] if "image" in f else input_dict["img_name"] + \
                                                                                            input_dict["seg_format"]
        img = Image.open(os.path.join(input_dict["root"], f, img_name))
        img.save(os.path.join(input_dict["root"], f + input_dict["out_suffix"], img_name))
        img.close()
        os.remove(os.path.join(input_dict["root"], f, img_name))

if __name__ == "__main__":
    root = "E:/Phd/projects/contrastive learning/2014_BOE_Srinivasan_2/Publication_Dataset/Test"
    image_format = ".tif"
    seg_format = ".tif"
    nproc = 1
    percentage = 0.3
    files = ["AMD",
             ]
    out_suffix = "_VAL"
    for f in files:
        mmcv.mkdir_or_exist(os.path.join(root, f + out_suffix))

    images = os.listdir(os.path.join(root, files[0]))
    images_idx = random.sample(range(0, len(images)), math.floor(len(images) * percentage))
    tasks = []
    for i in images_idx:
        img_name = images[i].split(image_format)[0]
        tasks.append({"root": root,
                      "files": files,
                      "img_name": img_name,
                      "image_format": image_format,
                      "seg_format": seg_format,
                      "out_suffix": out_suffix})

    if nproc > 1:  # handle it in parallel or not
        mmcv.track_parallel_progress(move_data, tasks, nproc)
    else:
        mmcv.track_progress(move_data, tasks)

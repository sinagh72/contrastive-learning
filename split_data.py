import math
import os
import mmcv
import random
from PIL import Image


def move_data(input_dict: dict):
    root = "C:/Users/Sina/Downloads/archive2/OCT2017_/train"
    for k, v in input_dict.items():
        for img_name in v:
            p1 = os.path.join(root, k.split("-")[0], img_name)
            p2 = os.path.join(root, k.split("-")[0] + "_TEST", img_name)
            img = Image.open(p1)
            img.save(p2)
            os.remove(p1)


if __name__ == "__main__":
    root = "C:/Users/Sina/Downloads/archive2/OCT2017_/train"
    image_format = ".jpeg"
    nproc = 1
    percentage = [0.9, 0.1]
    files = ["CNV",
             "DME",
             "DRUSEN",
             "NORMAL"
             ]
    test_suffix = "_TEST"
    img_names = {}
    for f in files:
        for i in os.listdir(os.path.join(root, f)):
            if i.split("-")[0] + "-" + i.split("-")[1] in img_names:
                img_names[i.split("-")[0] + "-" + i.split("-")[1]] += [i]
            else:
                img_names[i.split("-")[0] + "-" + i.split("-")[1]] = [i]
        mmcv.mkdir_or_exist(os.path.join(root, f + test_suffix))
    data = {"train": [], "test": []}
    for key in img_names:
        choice = random.choices(["train", "test"], weights=percentage, k=1)[0]
        data[choice].append({key: img_names[key]})
    print(len(data["train"]))
    print(len(data["test"]))
    # images = os.listdir(os.path.join(root, files[0]))
    # images_idx = random.sample(range(0, len(images)), math.floor(len(images) * percentage))
    # tasks = []
    # for i in images_idx:
    #     img_name = images[i].split(image_format)[0]
    #     tasks.append({"root": root,
    #                   "files": files,
    #                   "img_name": img_name,
    #                   "image_format": image_format,
    #                   "out_suffix": out_suffix})
    #
    if nproc > 1:  # handle it in parallel or not
        mmcv.track_parallel_progress(move_data, data["test"], nproc)
    else:
        mmcv.track_progress(move_data, data["test"])

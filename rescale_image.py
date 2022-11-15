"""
Created on 28/9/2022

@author: Sina Gholami
"""
import os
import mmcv
# from split_image.split import split, reverse_split
import math
from PIL import Image


class Rescaling(object):
    """
    BICUBIC = 3
    BILINEAR = 2
    BOX = 4
    HAMMING = 5
    LANCZOS = 1
    NEAREST = 0
    """

    def __init__(self, output_size, sampling_method=Image.Resampling.LANCZOS):
        # sampling method
        self.sampling_method = sampling_method
        self.output_size = output_size

    def __call__(self, img):
        return img.resize((self.output_size, self.output_size), self.sampling_method)


def rescale(input_dic: dict):
    img = Image.open(input_dic['input_img'])
    img = input_dic['rescaling'](img)
    img.save(input_dic['save_path'])


def main(source_path: str,
         target_size: int = 512,
         nproc=4):
    # iterating over the image source folder and read the images
    # define the rescaling function
    rescaling = Rescaling(output_size=target_size)
    img_filename_list = os.listdir(source_path)
    imgs = []
    for img_file in img_filename_list:
        root = os.path.join(source_path, img_file, "TIFFs", "8bitTIFFs")
        mmcv.mkdir_or_exist(root + "_128")
        imgs += [{"input_img": os.path.join(root, name),
                  "save_path": os.path.join(root + "_128", name), 'rescaling': rescaling}
                 for name in os.listdir(root)]

    # check whether the segmentation folder is specified by the user or not
    if nproc > 1:  # handle it in parallel or not
        mmcv.track_parallel_progress(rescale, imgs, nproc)
    else:
        mmcv.track_progress(rescale, imgs)


if __name__ == "__main__":
    source_path = "./2014_BOE_Srinivasan_2/Publication_Dataset/"
    main(source_path,
         target_size=128,
         nproc=4
         )

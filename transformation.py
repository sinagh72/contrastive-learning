import numbers

import PIL.Image
import torch
from PIL import ImageOps
from numpy import random
from torchvision.transforms import transforms as T
from torchvision.transforms import InterpolationMode


# class ContrastiveTransformations(object):
#
#     def __init__(self, transform_function, n_views=2):
#         self.transformations = transform_function
#         self.n_views = n_views
#
#     def __call__(self, x, apply_views: bool = True):
#         if apply_views:
#             return [self.transformations(x) for _ in range(self.n_views)]
#         return self.transformations(x)


def train_transformation():
    return T.Compose([T.Resize((128, 128), InterpolationMode.BICUBIC),
                      T.RandomHorizontalFlip(p=0.25),
                      T.RandomRotation(degrees=45),
                      T.RandomPerspective(distortion_scale=0.5, p=0.25),
                      T.RandomApply([
                          T.ColorJitter(brightness=0.2,
                                        contrast=0.2,
                                        saturation=0.2,
                                        hue=0.1),
                          T.GaussianBlur(kernel_size=3),
                          T.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3),
                                         scale=(0.5, 0.75)),
                          T.ElasticTransform(alpha=(50.0, 250.0), sigma=(5.0, 10.0))
                      ], p=0.25),
                      T.RandomGrayscale(p=0.25),
                      T.Grayscale(3),
                      T.ToTensor(),
                      T.Normalize((0.5,), (0.5,)),
                      # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                      ])


def train_transformation2():
    return T.Compose([
        T.Resize((128, 128), InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.25),
        T.RandomRotation(degrees=45),
        T.Grayscale(3),
        T.GaussianBlur(kernel_size=5),
        T.ColorJitter(brightness=0.4,
                      contrast=0.4,
                      saturation=0.4,
                      hue=0.2),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
        # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    ])


class RandomCrop(object):
    """
    Take a random crop from the image.
    First the image or crop size may need to be adjusted if the incoming image
    is too small...
    If the image is smaller than the crop, then:
         the image is padded up to the size of the crop
         unless 'nopad', in which case the crop size is shrunk to fit the image
    A random crop is taken such that the crop fits within the image.
    If a centroid is passed in, the crop must intersect the centroid.
    """

    def __init__(self, crop_size=(400, 400), resize=(128, 128), nopad=True):

        # if isinstance(crop_size, numbers.Number):
        #     self.size = (int(crop_size), int(crop_size))
        # else:
        #     self.size = crop_size
        self.crop_size = crop_size
        self.resize = resize
        self.nopad = nopad
        self.pad_color = (0, 0, 0)

    def __call__(self, img, centroid=None):
        w, h = img.size
        # ASSUME H, W
        th, tw = self.crop_size
        if w == tw and h == th:
            return img

        if self.nopad:
            if th > h or tw > w:
                # Instead of padding, adjust crop size to the shorter edge of image.
                shorter_side = min(w, h)
                th, tw = shorter_side, shorter_side
        else:
            # Check if we need to pad img to fit for crop_size.
            if th > h:
                pad_h = (th - h) // 2 + 1
            else:
                pad_h = 0
            if tw > w:
                pad_w = (tw - w) // 2 + 1
            else:
                pad_w = 0
            border = (pad_w, pad_h, pad_w, pad_h)
            if pad_h or pad_w:
                img = ImageOps.expand(img, border=border, fill=self.pad_color)
                w, h = img.size

        if centroid is not None:
            # Need to insure that centroid is covered by crop and that crop
            # sits fully within the image
            c_x, c_y = centroid
            max_x = w - tw
            max_y = h - th
            x1 = random.randint(c_x - tw, c_x)
            x1 = min(max_x, max(0, x1))
            y1 = random.randint(c_y - th, c_y)
            y1 = min(max_y, max(0, y1))
        else:
            if w == tw:
                x1 = 0
            else:
                x1 = random.randint(0, w - tw)
            if h == th:
                y1 = 0
            else:
                y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)).resize(size=self.resize, resample=PIL.Image.LANCZOS)


def train_transformation3():
    return T.Compose([
        RandomCrop(crop_size=(400, 400), resize=(128, 128)),
        T.Resize((128, 128), InterpolationMode.BICUBIC),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])


def train_transformation4():
    return T.Compose([
        RandomCrop(crop_size=(400, 400), resize=(128, 128)),
        T.RandomHorizontalFlip(),  # probability = 0.5
        T.RandomVerticalFlip(),  # probability = 0.5
        T.RandomRotation(degrees=30),  # probability = 0.5
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5)], p=0.8),
        T.GaussianBlur(kernel_size=9),
        T.Grayscale(3),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
    ])


def representation_transform(img):
    transform = T.Compose([T.RandomHorizontalFlip(p=0.5),
                           T.RandomRotation(degrees=45),
                           T.RandomPerspective(distortion_scale=0.5, p=0.5),
                           T.RandomGrayscale(p=0.2),
                           T.GaussianBlur(kernel_size=9),
                           T.Grayscale(1),
                           ])
    return transform(img.copy())

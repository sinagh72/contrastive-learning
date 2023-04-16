import torch
from torchvision.transforms import transforms as T
from torchvision.transforms import InterpolationMode


class ContrastiveTransformations(object):

    def __init__(self, transform_function, n_views=2):
        self.transformations = transform_function
        self.n_views = n_views

    def __call__(self, x, apply_views: bool = True):
        if apply_views:
            return [self.transformations(x) for _ in range(self.n_views)]
        return self.transformations(x)


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


def representation_transform(img):
    transform = T.Compose([T.RandomHorizontalFlip(p=0.5),
                           T.RandomRotation(degrees=45),
                           T.RandomPerspective(distortion_scale=0.5, p=0.5),
                           T.RandomGrayscale(p=0.2),
                           T.GaussianBlur(kernel_size=9),
                           T.Grayscale(1),
                           ])
    return transform(img.copy())

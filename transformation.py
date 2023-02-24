import torch
from torchvision.transforms import transforms, InterpolationMode


class ContrastiveTransformations(object):

    def __init__(self, transformations, n_views=2):
        self.transformations = transformations
        self.n_views = n_views

    def __call__(self, x, apply_views: bool = True):
        if apply_views:
            return [self.transformations(x) for _ in range(self.n_views)]
        return self.transformations(x)


def train_aug(img):
    # transform = transforms.Compose([
    #     transforms.Resize((128, 128), InterpolationMode.BICUBIC),
    #     transforms.RandomApply(torch.nn.ModuleList([transforms.ElasticTransform(alpha=(28.0, 30.0),
    #                                                                             sigma=(3.5, 4.0))]), p=0.3),
    #     transforms.RandomAffine(degrees=4.6, scale=(0.98, 1.02), translate=(0.03, 0.03)),
    #     transforms.RandomHorizontalFlip(p=0.25),
    #     transforms.Grayscale(3),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    transform = transforms.Compose([transforms.Resize((128, 128), InterpolationMode.BICUBIC),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=45),
                                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                    # transforms.RandomResizedCrop(size=128, scale=(0.25, 0.75),
                                    #                              interpolation=InterpolationMode.BICUBIC),
                                    transforms.RandomApply([
                                        transforms.ColorJitter(brightness=0.5,
                                                               contrast=0.5,
                                                               saturation=0.5,
                                                               hue=0.1),
                                        transforms.GaussianBlur(kernel_size=3),
                                    ], p=0.8),
                                    transforms.RandomGrayscale(p=0.2),

                                    transforms.Grayscale(3),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    # transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
                                    ])
    img = transform(img.copy())
    return img


def representation_transform(img):
    transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=45),
                                    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                    transforms.RandomGrayscale(p=0.2),
                                    transforms.GaussianBlur(kernel_size=9),
                                    transforms.Grayscale(1),
                                    ])
    return transform(img.copy())

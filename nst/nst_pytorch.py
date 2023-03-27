import os
from pathlib import Path

import numpy as np
import tables
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models
from dotenv import load_dotenv
from numpy import random
from tqdm import tqdm

from OCT_dataset import OCTDataset, get_kaggle_imgs


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.mean = mean.clone().detach().requires_grad_(True)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.std = std.clone().requires_grad_(True)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers,
                               style_layers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization).to(device)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, content_layers,
                       style_layers, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img, content_layers, style_layers)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


unloader = transforms.ToPILImage()


def check_img_validity(style_path, styles, num_retries=5):
    for attempt_no in range(num_retries):
        try:
            return Image.open(style_path).convert('RGB')
        except Exception as error:
            style_path = random.choice(styles, 1)[0]
            if attempt_no < (num_retries - 1):
                print(error)
            else:
                raise error

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def stylize_dataset_multiple(dataset, style_dir, out_file, style_weight=1, content_weight=1, save_size=256,
                             style_views=10, num_steps=300,
                             save_sample=False):
    # collect style files
    style_dir = Path(style_dir)
    extensions = ['png', 'jpeg', 'jpg']
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory'
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))

    import copy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # desired size of the output image
    imsize = save_size, save_size
    to_gray_scale = transforms.Grayscale(3)

    loader_rgb = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    loader_oct = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        to_gray_scale,
        transforms.ToTensor()])  # transform it into a torch tensor

    def image_loader(np_img, loader):
        # fake batch dimension required to fit network's input dimensions
        image = loader(np_img).unsqueeze(0)
        return image.to(device, torch.float)

    vgg = models.vgg19(weights="VGG19_Weights.DEFAULT").features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # decoder.eval()
    # vgg.eval()

    # decoder.load_state_dict(torch.load('models/decoder.pth'))
    # vgg.load_state_dict(torch.load('models/vgg_normalised.t7'))
    # vgg = nn.Sequential(*list(vgg.children())[:31])

    # vgg.to(device)
    # decoder.to(device)

    # crop = 0
    # content_tf = input_transform(content_size, crop)
    # style_tf = input_transform(style_size, 0)

    # hdf5_file = tables.open_file(out_path, mode='w')
    # data_shape = (0, save_size, save_size, 3)
    # img_dtype = tables.UInt8Atom()
    # storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)

    # tile_paths = []
    # tile_names = []
    generated = os.listdir("../data/nst_data_balanced")
    save_count = 1000
    # actual style transfer as in AdaIN
    for idx in tqdm(range(len(dataset))):
        img_name = dataset[idx]["img_name"]
        img_path = dataset[idx]["img_path"]

        # try:
        content_info = dataset[idx]["img"]

        if f"{img_name[:-5]}_{0}.jpg" in generated and f"{img_name[:-5]}_{1}.jpg" in generated and \
                f"{img_name[:-5]}_{2}.jpg" in generated:
            continue
        print(img_name[:-5])
        for i, style_path in enumerate(random.choice(styles, style_views, replace=False)):
            style_img = check_img_validity(style_path, styles)
            # style_img = Image.open(style_path).convert('RGB')

            # if save_count != 0:
            #     style_img.save(f"../data/sample_nst/style_img_{idx}_{i}.jpg")
            content_img = image_loader(content_info.copy(), loader_oct)
            style_img = image_loader(style_img, loader_rgb)
            input_img = content_img.clone()
            output = run_style_transfer(cnn=vgg, normalization_mean=cnn_normalization_mean[:, None, None],
                                        normalization_std=cnn_normalization_std[:, None, None],
                                        content_img=content_img, style_img=style_img, input_img=input_img,
                                        content_layers=content_layers_default, num_steps=num_steps,
                                        content_weight=content_weight, style_weight=style_weight,
                                        style_layers=style_layers_default)
            output = output.detach().squeeze(0)
            output = to_gray_scale(output)
            # if save_count != 0:
            unloader(output).save(f"../data/{out_file}/{img_name[:-5]}_{i}.jpg")
            #     save_count -= 1
            # unloader(output).show()

            # print(output.permute(1, 2, 0)[None].numpy().shape)
            # storage.append(output.permute(1, 2, 0)[None].numpy())
            # tilename = img_name[:-5] + '_stylized_' + os.path.basename(style_path)[:-4] + img_name[-5:]
            # tile_names.append(tilename)
            # tile_paths.append(img_path.replace("../", ""))
        content_info.close()
        # except Exception as err:
        #     print(f'skipped stylization of {fname} because of the following error; {err})')
        #     skipped_imgs.append(fname)
        #     continue

    # if len(skipped_imgs) > 0:
    #     with open(os.path.join(os.path.dirname(out_path), 'skipped_imgs.txt'), 'w') as f:
    #         for item in skipped_imgs:
    #             f.write("%s\n" % item)

    # hdf5_file.create_array(hdf5_file.root, 'paths', tile_paths)
    # hdf5_file.create_array(hdf5_file.root, 'names', tile_names)
    # hdf5_file.close()


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "8"

    # args = parser.parse_args()
    alpha = 0.15
    style_views = 3
    save_size = 128

    classes = [("NORMAL", 0),
               ("AMD", 1),
               ("DME", 2)]

    # dataset = OCTDataset(data_root=content_path,
    #                      img_type="RGB",
    #                      transform=None,
    #                      classes=classes,
    #                      mode="test",
    #                      dataset_func=get_duke_imgs,
    #                      ignore_folders=[],
    #                      sub_folders_name="TIFFs/8bitTIFFs",
    #                      )
    load_dotenv(dotenv_path="../data/.env")
    style_path = os.getenv('STYLES_PATH')
    DATASET_PATH = os.getenv('KAGGLE_FULL_DATASET_PATH')
    out_file = "nst_data_full"

    dataset = OCTDataset(data_root=DATASET_PATH,
                         img_type="RGB",
                         transform=None,
                         classes=classes,
                         mode="test",
                         dataset_func=get_kaggle_imgs,
                         )
    stylize_dataset_multiple(dataset=dataset, style_dir=style_path, out_file=out_file,
                             style_weight=2000, content_weight=1, num_steps=50,
                             save_size=save_size, style_views=style_views, save_sample=False)


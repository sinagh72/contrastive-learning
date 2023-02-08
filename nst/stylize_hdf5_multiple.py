# https://github.com/bethgelab/stylize-datasets
import os
import sys

from tqdm import tqdm

from OCT_dataset import representation_transform

sys.path.append(os.path.dirname(__file__))
import argparse
import random
import numpy as np
from PIL import Image
from pathlib import Path
import tables
import torch
import torch.nn as nn
import torchvision.transforms
from function import adaptive_instance_normalization
import net


def input_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(torchvision.transforms.Resize(size))
    if crop != 0:
        transform_list.append(torchvision.transforms.CenterCrop(crop))
    transform_list.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content, style, alpha=1.0):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)
    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


def stylize_dataset_multiple(dataset, style_dir, out_path, alpha=1., content_size=1024,
                             style_size=256, save_size=256, style_views=10):
    # collect style files
    style_dir = Path(style_dir)
    style_dir = style_dir.resolve()
    extensions = ['png', 'jpeg', 'jpg']
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))

    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    crop = 0
    content_tf = input_transform(content_size, crop)
    style_tf = input_transform(style_size, 0)

    hdf5_file = tables.open_file(out_path, mode='w')
    data_shape = (0, save_size, save_size, 3)
    img_dtype = tables.UInt8Atom()
    storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)

    # disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []

    tile_paths = []
    tile_names = []

    # actual style transfer as in AdaIN
    for idx in tqdm(range(len(dataset))):
        img_name = dataset[idx]["img_name"]
        img_path = dataset[idx]["img_path"]
        # try:
        content_img = dataset[idx]["img"]
        for style_path in random.sample(styles, style_views):
            style_img = Image.open(style_path).convert('RGB')
            # showing purposes
            style_img.show("Style image")
            org_img = Image.fromarray(content_img)
            org_img.show("Original Image")
            representation_transform(org_img).show("regular augmentation")


            content = content_tf(content_img)
            style = style_tf(style_img)
            style = style.to(device).unsqueeze(0)
            content = content.to(device).unsqueeze(0)
            with torch.no_grad():
                output = style_transfer(vgg, decoder, content, style, alpha)
            output = output.cpu().squeeze_(0)
            output_img = torchvision.transforms.ToPILImage()(output)

            output_img.show("nst Image")
            representation_transform(output_img).show("nst regular augmentation")

            output_img = output_img.resize((save_size, save_size), Image.LANCZOS)
            output = np.array(output_img)

            storage.append(output[None])
            tilename = img_name[:-5] + '_stylized_' + os.path.basename(style_path)[:-4] + img_name[-5:]
            tile_names.append(tilename)
            tile_paths.append(img_path.replace("../", ""))
            style_img.close()
        content_img.close()

        # except Exception as err:
        #     print(f'skipped stylization of {fname} because of the following error; {err})')
        #     skipped_imgs.append(fname)
        #     continue

    # if len(skipped_imgs) > 0:
    #     with open(os.path.join(os.path.dirname(out_path), 'skipped_imgs.txt'), 'w') as f:
    #         for item in skipped_imgs:
    #             f.write("%s\n" % item)

    hdf5_file.create_array(hdf5_file.root, 'paths', tile_paths)
    hdf5_file.create_array(hdf5_file.root, 'names', tile_names)
    hdf5_file.close()


def stylize_hdf5_multiple(contents, style_dir, out_path, ids, fnames, labels, alpha=1., content_size=1024,
                          style_size=256, save_size=256, num_styles=10):
    # collect style files
    style_dir = Path(style_dir)
    style_dir = style_dir.resolve()
    extensions = ['png', 'jpeg', 'jpg']
    styles = []
    for ext in extensions:
        styles += list(style_dir.rglob('*.' + ext))

    assert len(styles) > 0, 'No images with specified extensions found in style directory' + style_dir
    styles = sorted(styles)
    print('Found %d style images in %s' % (len(styles), style_dir))

    decoder = net.decoder
    vgg = net.vgg

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    decoder.eval()
    vgg.eval()

    decoder.load_state_dict(torch.load('models/decoder.pth'))
    vgg.load_state_dict(torch.load('models/vgg_normalised.pth'))
    vgg = nn.Sequential(*list(vgg.children())[:31])

    vgg.to(device)
    decoder.to(device)

    crop = 0
    content_tf = input_transform(content_size, crop)
    style_tf = input_transform(style_size, 0)

    hdf5_file = tables.open_file(out_path, mode='w')
    data_shape = (0, save_size, save_size, 3)
    img_dtype = tables.UInt8Atom()
    storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)

    # disable decompression bomb errors
    Image.MAX_IMAGE_PIXELS = None
    skipped_imgs = []

    tile_ids = []
    tile_fnames = []
    tile_labels = []

    # actual style transfer as in AdaIN
    for idx in range(len(contents)):
        try:
            content_img = Image.fromarray(contents[idx, :, :, :]).convert('RGB')
            for style_path in random.sample(styles, num_styles):
                style_img = Image.open(style_path).convert('RGB')

                content = content_tf(content_img)
                style = style_tf(style_img)
                style = style.to(device).unsqueeze(0)
                content = content.to(device).unsqueeze(0)
                with torch.no_grad():
                    output = style_transfer(vgg, decoder, content, style, alpha)
                output = output.cpu().squeeze_(0)
                output_img = torchvision.transforms.ToPILImage()(output)
                output_img = output_img.resize((save_size, save_size), Image.LANCZOS)
                output = np.array(output_img)

                storage.append(output[None])

                tilename = fnames[idx][:-4] + '_stylized_' + os.path.basename(style_path)[:-4] + fnames[idx][-4:]
                tile_fnames.append(tilename)
                tile_ids.append(ids[idx])
                tile_labels.append(labels[idx])
                style_img.close()
            content_img.close()

        except Exception as err:
            print(f'skipped stylization of {fnames[idx]} because of the following error; {err})')
            skipped_imgs.append(fnames[idx])
            continue

    if len(skipped_imgs) > 0:
        with open(os.path.join(os.path.dirname(out_path), 'skipped_imgs.txt'), 'w') as f:
            for item in skipped_imgs:
                f.write("%s\n" % item)

    hdf5_file.create_array(hdf5_file.root, 'ids', tile_ids)
    hdf5_file.create_array(hdf5_file.root, 'fnames', tile_fnames)
    hdf5_file.create_array(hdf5_file.root, 'labels', tile_labels)
    hdf5_file.close()

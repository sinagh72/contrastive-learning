import argparse
import h5py
from PIL import ImageFile
# from stylize_hdf5_single import stylize_hdf5_single
import os

from OCT_dataset import OCTDataset, get_kaggle_imgs, get_duke_imgs
# to generate multiple stylized images per each content image, comment out above and uncomment below
from stylize_hdf5_multiple import stylize_hdf5_multiple, stylize_dataset_multiple

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser()
parser.add_argument('--content-path', type=str,
                    help='path to the content images in hdf5 format')
parser.add_argument('--style-dir', type=str,
                    help='directory path to a batch of style images')
parser.add_argument('--out-path', type=str,
                    help='path to save the stylized dataset in hdf5 format')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='the weight that controls the degree of \
                          stylization, should be between 0 and 1')
parser.add_argument('--content-size', type=int, default=1024,
                    help='new (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style-size', type=int, default=256,
                    help='new (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--save-size', type=int, default=256,
                    help='output size for the stylized image')
# to generate multiple stylized images per each content image, uncomment below
# parser.add_argument('--num-styles', type=int, default=1, help='number of styles to \
#                         create for each image (default: 1)')

if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # args = parser.parse_args()
    content_path = "../data/2014_BOE_Srinivasan_2/Publication_Dataset/original data"
    style_dir = "/sina/train/"
    out_path = "../data/nst_full.hdf5"
    alpha = 0.5
    style_views = 3
    content_size = 512
    style_size = 256
    save_size = 256

    classes = [("NORMAL", 0),
               ("AMD", 1),
               ("DME", 2)]

    dataset = OCTDataset(data_root=content_path,
                         img_type="RGB",
                         transform=None,
                         classes=classes,
                         mode="test",
                         dataset_func=get_duke_imgs,
                         ignore_folders=[],
                         sub_folders_name="TIFFs/8bitTIFFs",
                         )

    # h5 = h5py.File(content_path, mode='r')
    # imgs = h5['img']
    # ids = [i.decode('UTF-8') for i in h5['ids']]
    # fnames = [i.decode('UTF-8') for i in h5['fnames']]
    # labels = [i for i in h5['labels']]

    # stylize_hdf5_single(contents=imgs, style_dir=args.style_dir, out_path=args.out_path, ids=ids, fnames=fnames, labels=labels, alpha=args.alpha, content_size=args.content_size, style_size=args.style_size, save_size=args.save_size)
    # to generate multiple stylized images per each content image, comment out above and uncomment below
    # stylize_hdf5_multiple(contents=imgs, style_dir=style_dir, out_path=out_path, ids=ids,
    #                       fnames=fnames, labels=labels, alpha=alpha, content_size=content_size,
    #                       style_size=style_size, save_size=save_size, num_styles=num_styles)

    stylize_dataset_multiple(dataset=dataset, style_dir=style_dir, out_path=out_path, alpha=alpha,
                             content_size=content_size,
                             style_size=style_size, save_size=save_size, style_views=style_views)

import h5py
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image
from OCT_dataset import OCTDataset, get_dfDict, ContrastiveTransformations, train_aug
from kaggle_compare import _to_three_channel

TEST_DATASET_PATH = "data/kaggle_dataset_full/"
img_transforms = transforms.Compose([transforms.Resize(size=(128, 128), interpolation=InterpolationMode.BICUBIC),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5,), (0.5,)),
                                     transforms.Lambda(_to_three_channel)])

classes = [("NORMAL", 0),
           ("AMD", 1),
           ("DME", 2)]

# dataset = OCTDataset(data_root=TEST_DATASET_PATH,
#                            transform=None,
#                            classes=classes,
#                            mode="test",
#                            img_type="RGB"
#                            )
h5 = h5py.File("./data/nst.hdf5")
images, df_dic, names = get_dfDict("./data/nst.hdf5")

trans = ContrastiveTransformations(n_views=2, transformations=train_aug)
l = [Image.fromarray(h5["img"][0]), Image.fromarray(h5["img"][0])]
res = trans(l)

# print(h5['img'])
# print([i.decode('UTF-8') for i in h5['fnames']])

"""
Find the difference between two folders
To check whether the segmentation (classification) and image folders have the same images
"""
import os


def diff(first_dir, second_dir, first_dir_img_format=".png", second_dir_img_format=".png", remove_diff=False):
    """
    Finds the images that common in both first_dir and second_dir based on their names disregarding their format.
     It is possible to remove those images that are ont inside their intersection.

    Args:
        -first_dir (str): path to the first directory
        - first_dir_img_format (str): the format of the images inside the first_dir
        -second_dir (str): path to the second directory
        - second_dir_img_format (str): the format of the images inside the second_dir
        -remove_diff (bool): whether to remove images that are not in their intersection
    """
    d1 = sorted(os.listdir(first_dir))
    d2 = sorted(os.listdir(second_dir))
    f1 = []
    for f in d1:
        f1.append(f.split(first_dir_img_format)[0])

    f2 = []
    for f in d2:
        f2.append(f.split(second_dir_img_format)[0])

    intersection = list(set(f2).intersection(set(f1)))
    print("intersection between two files", len(intersection))

    if remove_diff:
        for f in intersection:
            os.remove(os.path.join(second_dir, f) + second_dir_img_format)

        # for f in intersection:
        #     os.remove(os.path.join(first_dir, f) + first_dir_img_format)


if __name__ == "__main__":
    diff(first_dir="./kaggle_dataset/test/NORMAL",
         second_dir="./kaggle_dataset/train/NORMAL",
         first_dir_img_format=".jpeg",
         second_dir_img_format=".jpeg",
         remove_diff=True)

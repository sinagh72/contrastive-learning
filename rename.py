import os


def change_name(dir_path: str):
    name = os.listdir(dir_path)
    for n in name:
        os.rename(os.path.join(dir_path, n), os.path.join(dir_path, n.replace(".png", "_nonreferral.png")))


if __name__ == "__main__":
    change_name("C:/Phd/projects/contrastive learning/data/DR data-reorganized/train/nonreferral")

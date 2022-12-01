import numpy as np

from OCT_dataset import OCTDataset, ContrastiveTransformations, train_aug

if __name__ == "__main__":
    N_VIEWS = 2
    CV = 5
    PATIENTS = 15
    cv_step = PATIENTS // CV

    idx = np.array(range(1, cv_step + 1))
    for i in range(CV):
        val_dataset = OCTDataset(data_root="./2014_BOE_Srinivasan_2/Publication_Dataset/original data",
                                 img_suffix='.tif',
                                 transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
                                 folders=idx)
        # print(set(np.array(range(1, PATIENTS + 1))) -set(choices))
        train_dataset = OCTDataset(data_root="./2014_BOE_Srinivasan_2/Publication_Dataset/original data",
                                   img_suffix='.tif',
                                   transform=ContrastiveTransformations(train_aug, n_views=N_VIEWS),
                                   folders=list(set(np.array(range(1, PATIENTS + 1))) - set(idx)))

        print(len(train_dataset) + len(val_dataset))
        choices += 3

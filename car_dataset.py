from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from img_preprocessing import *
from config import Config


class CarDataset(Dataset):

    def __init__(self, datafram, root_dir, training=True, transform=None):
        super(CarDataset, self).__init__()
        self.df = datafram
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1

        # Read image
        if Config.USE_MASK:
            if self.root_dir == Config.DATA_PATH + 'train_images/{}.jpg':
                mask_path = Config.DATA_PATH + 'MaskTrain/{}.jpg'
            else:
                mask_path = Config.DATA_PATH + 'MaskTest/{}.jpg'
            img_name = mask_path.format(idx)
            img0 = imread(img_name, True)
        else:
            img0 = imread(img_name, True)

        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)

        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr]


def get_data_set():
    train = pd.read_csv(Config.DATA_PATH + 'train.csv')
    test = pd.read_csv(Config.DATA_PATH + 'sample_submission.csv')

    df_train, df_valid = train_test_split(train, test_size=0.05, random_state=42)
    df_test = test
    return df_train, df_valid, df_test


def get_data_loader():
    train_images_dir = Config.DATA_PATH + 'train_images/{}.jpg'
    test_images_dir = Config.DATA_PATH + 'test_images/{}.jpg'

    df_train, df_valid, df_test = get_data_set()

    # Create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, training=True)
    dev_dataset = CarDataset(df_valid, train_images_dir, training=False)
    test_dataset = CarDataset(df_test, test_images_dir, training=False)

    # Create data generators - they will produce batches
    train_loader = DataLoader(dataset=train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=4)
    valid_loader = DataLoader(dataset=dev_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                             num_workers=0)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    train_dataset = get_data_loader()[0].dataset

    img, mask, regr = train_dataset[0]
    # for i in train_loader:
    #     print(i)
    plt.figure(figsize=(16, 16))
    plt.imshow(np.rollaxis(img, 0, 3))
    plt.show()

    plt.figure(figsize=(16, 16))
    plt.imshow(mask)
    plt.show()

    plt.figure(figsize=(16, 16))
    plt.imshow(regr[-2])
    plt.show()

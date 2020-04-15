
import os
import torch
import argparse

import numpy as np
import pandas as pd
from time import time
from torch.autograd import Variable

from glob import glob
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


import sys
sys.path.append('../')
from src.tilenet import make_tilenet
from src.resnet import ResNet18

from helpers import load_meta, resize_bands, load_patch

def get_labels(df):

    # Get unique labels and make dictionary
    uq      = list(df['labels'].to_list())
    uq      = set([item for sublist in uq for item in sublist])
    y_dict  = dict(zip(uq, range(len(uq))))

    # Get labels
    y = np.zeros((len(metas), len(y_dict)))
    for i, row in metas.iterrows():
        y[i, [y_dict[l] for l in row.labels]] = 1

    return y, y_dict


class BigEarthNetDL(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df):
        self.list_ids = df

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Get tile
        tile = load_patch(os.path.join(self.list_ids.iloc[index].patch_dir))

        # randomly sample 50x50 section of the tile
        x_idx = np.random.randint(low=0, high=120-50)
        y_idx = np.random.randint(low=0, high=120-50)
        tile = tile[x_idx:x_idx+50,y_idx:y_idx+50,:]

        # Rearrange to PyTorch order
        tile = np.moveaxis(tile, -1, 0)
        tile = np.expand_dims(tile, axis=0)

        # Scale to [0, 1]
        tile = tile / 255
        # Load data and get label

        return tile


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_dir', default='', type=str, help='Path to bigearthnet triplets')
    parser.add_argument('--model_dir', default='', type=str, help='Path to model saving')
    parser.add_argument('--n_tiles', default=50000, type=int, help='Number to sample')
    parser.add_argument('--batch_size', default=10, type=int, help='')
    parser.add_argument('--in_channels', default=12, type=int, help='12 for landsat')
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--n_triplets', default=20000, type=int, help='size of dataset')
    parser.add_argument('--z_dim', default=512, type=int, help='dims of embedding space')
    parser.add_argument('--epochs', default=10, type=int, help='')
    parser.add_argument('--lr', default=0.01, type=float, help='')
    parser.add_argument('--margin', default=10, type=int, help='learning rate')
    parser.add_argument('--l2', default=0.01, type=float, help='learning rate')
    config = parser.parse_args()

    # Setting up model
    cuda = torch.cuda.is_available()
    tilenet = make_tilenet(in_channels=config.in_channels, z_dim=config.z_dim)
    if cuda: tilenet.cuda()

    # Load parameters
    checkpoint = torch.load(config.model_dir)
    tilenet.load_state_dict(checkpoint)
    tilenet.eval()


    # Get tile metadata
    tile_dirs = glob(os.path.join(config.tile_dir, '*'))
    metas = pd.DataFrame([load_meta(t) for t in tqdm(config.tile_dirs)])[0:config.n_tiles]
    if config.shuffle==True:
        metas = metas.sample(frac=1)

    # Get labels
    y, y_dict = get_labels(metas)
    print(len(y_dict), "unique labels for", config.n_tiles, "tiles")

    dataset = BigEarthNetDL(metas)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle=False, num_workers=4)
    t0 = time()
    X = np.zeros((config.n_tiles, config.z_dim))
    for idx, tile in enumerate(dataloader):

        tile = torch.from_numpy(tile).float()
        tile = Variable(tile)
        if cuda: tile = tile.cuda()
        z = tilenet.encode(tile)
        if cuda: z = z.cpu()
        z = z.data.numpy()
        X[idx,:] = z
    t1 = time()
    print(X.shape)
    print(y.shape)
    print('Embedded {} tiles: {:0.3f}s'.format(config.n_tiles, t1-t0))

    #
    # # Embed tiles
    # t0 = time()
    # X = np.zeros((config.n_tiles, config.z_dim))
    # for idx, row in tqdm(metas.iterrows(), total=metas.shape[0]):
    # #for idx, row in metas.iterrows():
    #     # Get tile
    #     tile = load_patch(os.path.join(row.patch_dir))
    #
    #     # randomly sample 50x50 section of the tile
    #     x_idx = np.random.randint(low=0, high=120-50)
    #     y_idx = np.random.randint(low=0, high=120-50)
    #     tile = tile[x_idx:x_idx+50,y_idx:y_idx+50,:]
    #
    #     # Rearrange to PyTorch order
    #     tile = np.moveaxis(tile, -1, 0)
    #     tile = np.expand_dims(tile, axis=0)
    #
    #     # Scale to [0, 1]
    #     tile = tile / 255
    #
    #     # Embed tile
    #     tile = torch.from_numpy(tile).float()
    #     tile = Variable(tile)
    #
    #     if cuda: tile = tile.cuda()
    #     z = tilenet.encode(tile)
    #     if cuda: z = z.cpu()
    #     z = z.data.numpy()
    #
    #     X[idx,:] = z
    # t1 = time()
    # print('Embedded {} tiles: {:0.3f}s'.format(config.n_tiles, t1-t0))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Splitting data and training RF classifer
    X_trn, X_val, y_trn, y_val = train_test_split(X, y, test_size=0.2)
    rf = RandomForestClassifier(n_estimators=1000,
                                max_depth=10,
                                max_features='sqrt',
                                random_state=1)
    rf.fit(X_trn, y_trn)
    print("AND EVAL", rf.eval(X_val, y_val))
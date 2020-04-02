import os
import argparse
import pickle

import numpy as np
import pandas as pd

import torch
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


def multihot(y, labs):
    y_hot = np.zeros((1, len(labs)))
    y_hot[:, y] = 1
    return y_hot

class BigEarthNetDL(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, labels):
        self.list_ids = df
        self.y = pickle.load(open(labels, 'rb'))
        self.labs = set(x for l in self.y.values() for x in l)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):

        # Get tile
        path = self.list_ids.iloc[index].patch_dir
        name = os.path.basename(path)
        tile = load_patch(path)

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
        y = multihot(self.y[name], self.labs)

        return tile, y


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_dir', default='', type=str, help='Path to bigearthnet triplets')
    parser.add_argument('--labels_dir', default='', type=str, help='Path to ben labels')
    parser.add_argument('--model_dir', default='', type=str, help='Path to model saving')
    parser.add_argument('--feats_dir', default='', type=str, help='Path to save features')
    parser.add_argument('--n_tiles', default=50000, type=int, help='Number to sample')
    parser.add_argument('--batch_size', default=10, type=int, help='')
    parser.add_argument('--in_channels', default=12, type=int, help='12 for landsat')
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--z_dim', default=512, type=int, help='dims of embedding space')
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
    metas = pd.DataFrame([load_meta(t) for t in tqdm(tile_dirs)])[0:config.n_tiles]
    if config.shuffle == True:
        metas = metas.sample(frac=1)

    # Initialized DataLoader
    dataset = BigEarthNetDL(metas, config.labels_dir)
    dataloader = DataLoader(dataset, batch_size = config.batch_size, shuffle=True, num_workers=4)

    # Iterate and produce embeddings
    t0 = time()
    y = []
    X = np.zeros((config.n_tiles, config.z_dim))
    for idx, (tile, lab) in enumerate(dataloader):
        #tile = torch.from_numpy(tile).float()
        tile = torch.squeeze(tile)
        tile = Variable(tile)
        if cuda: tile = tile.cuda()
        z = tilenet.encode(tile)
        if cuda: z = z.cpu()
        z = z.data.numpy()
        X[idx,:] = z
        y.append(lab)
    t1 = time()
    y = np.stack(y, axis=0)
    model_name = os.path.basename(config.model_dir)

    # Save Features
    np.save(os.path.join(config.feats_dir, f'X_{model_name}.npy'), X)
    np.save(os.path.join(config.feats_dir, f'y_{model_name}.npy'), y)





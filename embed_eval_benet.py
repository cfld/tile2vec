import os
import argparse
import pickle
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd

from time import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import metrics
from sklearn.model_selection import train_test_split

import sys
sys.path.append('../')
from src.tilenet import make_tilenet
from src.resnet import ResNet18
from src.data_utils import clip_and_scale_image
from helpers import load_meta, resize_bands, load_patch


def multihot(y, labs):
    y_hot = np.zeros((1, len(labs)))
    y_hot[:, y] = 1
    return y_hot




class BigEarthNetDL(Dataset):

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
        # x_idx = np.random.randint(low=0, high=120-50)
        # y_idx = np.random.randint(low=0, high=120-50)
        # tile = tile[x_idx:x_idx+50,y_idx:y_idx+50,:]
        tile = tile[35:85,35:85,:]
        # Scale
        tile = tile.clip_and_scale_image(tile,img_type='landsat')

        # Rearrange to PyTorch order
        tile = np.moveaxis(tile, -1, 0)
        tile = np.expand_dims(tile, axis=0)

         # Load data and get label
        y = multihot(self.y[name], self.labs)

        return tile, y


def to_numpy(x):
    return x.detach().cpu().numpy()


class Flatten(nn.Module):
    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class MLPClassifier(nn.Module):
    def __init__(self, n_input, n_classes, n_hidden=512, p=0.1):
        super().__init__()

        self.n_input = n_input
        self.n_classes = n_classes
        self.n_hidden = n_hidden

        if n_hidden is None:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_classes, bias=True)
            )
        else:
            self.block_forward = nn.Sequential(
                Flatten(),
                nn.Dropout(p=p),
                nn.Linear(n_input, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_hidden, bias=False),
                nn.BatchNorm1d(n_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=p),
                nn.Linear(n_hidden, n_classes, bias=True)
            )

    def forward(self, x):
        return self.block_forward(x)


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
    metas = pd.DataFrame([load_meta(t) for t in tqdm(tile_dirs)])#[0:config.n_tiles]
    if config.shuffle == True:
        metas = metas.sample(frac=1)

    # Initialized DataLoader
    dataset     = BigEarthNetDL(metas, config.labels_dir)
    dataloader  = DataLoader(dataset,
                             batch_size     = config.batch_size,
                             shuffle        = True,
                             num_workers    = 4)

    # Iterate and produce embeddings
    t0 = time()
    y = []
    X = []
    for idx, (tile, lab) in enumerate(dataloader):
        tile = torch.squeeze(tile.float())
        tile = Variable(tile)
        if cuda: tile = tile.cuda()
        z = tilenet.encode(tile)
        if cuda: z = z.cpu()
        z = z.data.numpy()
        X.append(z)
        y.append(lab.squeeze().data.numpy())
    t1 = time()
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=3_000, random_state=123)
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_valid = torch.FloatTensor(X_valid)

    model = MLPClassifier(n_classes=19, n_input=X.shape[1])
    model = model.cuda()

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=32,
                              shuffle=True,
                              pin_memory=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        _ = model.train()
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)
            loss = F.binary_cross_entropy_with_logits(out, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

        _ = model.eval()
        z = model(X_valid.cuda())
        p_valid = to_numpy(z)
        auc_valid = [metrics.roc_auc_score(y, p) for y, p in zip(y_valid.T, p_valid.T)]
        print(np.mean(auc_valid))






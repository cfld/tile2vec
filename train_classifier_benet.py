import numpy as np
import pandas as pd
import os
import torch
from time import time
from torch.autograd import Variable
import json
import sys
sys.path.append('../')
from src.tilenet import make_tilenet
from src.resnet import ResNet18

from glob import glob
from tqdm import tqdm

## Generate mosaic functions
def load_meta(patch_dir):
    patch_name = os.path.basename(patch_dir)
    meta_path = os.path.join(patch_dir, patch_name + '_labels_metadata.json')

    meta = json.load(open(meta_path))
    del meta['projection']
    del meta['coordinates']
    meta.update({
        "patch_name": patch_name,
        "patch_dir": patch_dir,
        "row": int(patch_name.split('_')[-1]),
        "col": int(patch_name.split('_')[-2])
    })

    return meta

# Setting up model
in_channels = 12
z_dim = 512
cuda = torch.cuda.is_available()
# tilenet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
# Use old model for now
tilenet = ResNet18()
if cuda: tilenet.cuda()

# Load parameters
#model_fn = '../models/naip_trained.ckpt'
model_fn = '/raid/users/ebarnett/tile2vec/models/TileNet_epoch10.ckpt'
checkpoint = torch.load(model_fn)
tilenet.load_state_dict(checkpoint)
tilenet.eval()

# Get data
tile_dir = '/raid/users/bjohnson/data/bigearthnet/bigearthnet/'

n_tiles = 1000

# TODO: Generate labels
tile_dirs = glob(os.path.join(tile_dir, '*'))
metas = pd.DataFrame([load_meta(t) for t in tqdm(tile_dirs)])
uq = metas.labels.unique()
uq = [item for sublist in uq for item in sublist]
y_dict = dict(zip(uq,range(len(uq))))
y = np.zeros((len(metas, len(y_dict.keys()))))
for i, row in metas.iterrows():
    y[i, y_dict[row.label()]] = 1


# Embed tiles
t0 = time()
X = np.zeros((n_tiles, z_dim))
for idx in range(n_tiles):
    tile = np.load(os.path.join(tile_dir, '{}tile.npy'.format(idx+1)))
    # randomly sample 50x50 section of the tile
    x_idx = np.random.randint(low=0, high=120-50)
    y_idx = np.random.randint(low=0, high=120-50)
    tile = tile[x_idx:x_idx+50,y_idx:y_idx+50,:]
    # Rearrange to PyTorch order
    tile = np.moveaxis(tile, -1, 0)
    tile = np.expand_dims(tile, axis=0)
    # Scale to [0, 1]
    tile = tile / 255
    # Embed tile
    tile = torch.from_numpy(tile).float()
    tile = Variable(tile)
    if cuda: tile = tile.cuda()
    z = tilenet.encode(tile)
    if cuda: z = z.cpu()
    z = z.data.numpy()
    X[idx,:] = z
t1 = time()
print('Embedded {} tiles: {:0.3f}s'.format(n_tiles, t1-t0))

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
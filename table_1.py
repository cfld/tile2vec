import sys
import os
import torch
from glob import glob
from torch import optim
from time import time
import numpy as np
from scipy import stats

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

tile2vec_dir = '/home/users/ebarnett/tile2vec'
sys.path.append('../')
sys.path.append(tile2vec_dir)
from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from src.tilenet import make_tilenet
from src.training import prep_triplets, train_triplet_epoch
from src.resnet import ResNet18

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset



class NAIP(Dataset):

    def __init__(self, paths):
        self.paths = []
        for p in paths:
            try:
                a = np.load(p)
                c, w, h = a.shape
                if w and h >= 50:
                    self.paths.append(p)
            except:
                continue


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        img = np.load(self.paths[index])
        c, w, h = img.shape
        img = img[:,int(w/2)-25:int(w/2)+25, int(h/2)-25:int(h/2)+25]
        y = stats.mode(img[4,:,:])[0][0][0]
        img = img[0:4,:,:]
        return img, y


model_dir = '/home/ebarnett/tile2vec/models/'
tile_dir = '/raid/users/ebarnett/tile2vec/tiles/'

# Environment stuff
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
cuda = torch.cuda.is_available()

# USING PRETRAINED MODEL
print("Using pretrained model")
# Setting up model
in_channels = 4
z_dim = 512
n_trials = 5
cuda = torch.cuda.is_available()
# tilenet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
# Use old model for now
tilenet = ResNet18()
if cuda: tilenet.cuda()

# GET DATA
states_dir = os.path.join('/home/bjohnson/projects/naip_collect/data/states')
states_paths = [i for i, f, r in os.walk(states_dir)][1::]

patches_paths = []
states_paths = states_paths[0:5]
for state in states_paths:
    for _, _, files in os.walk(state):
        for f in files:
            if f.endswith(".npy"):
                patches_paths.append(os.path.join(state, f))

print(len(patches_paths))


# Initialized DataLoader
dataset = NAIP(patches_paths)
dataloader = DataLoader(dataset,
                        batch_size=64,
                        shuffle=True,
                        num_workers=4)

print("done with dataloader")
# Load parameters
model_fn = os.path.join(model_dir, 'naip_trained.ckpt')
checkpoint = torch.load(model_fn)
tilenet.load_state_dict(checkpoint)
tilenet.eval()

print("training...")
y = []
X = []
for idx, (tile, lab) in enumerate(dataloader):
    tile = torch.squeeze(tile.float())
    tile = Variable(tile)
    if cuda: tile = tile.cuda()
    z = tilenet.encode(tile)
    if cuda: z = z.cpu()
    X.append(z.data.numpy())
    y.append(lab.data.numpy())
t1 = time()
X = np.concatenate(X, axis=0)
y = np.concatenate(y, axis=0)

for n_tiles in [1000, 10000]:
    # Embed tiles
    y = LabelEncoder().fit_transform(y)

    accs_rf = np.zeros((n_trials,))
    accs_log = np.zeros((n_trials,))
    accs_nn = np.zeros((n_trials,))

    for i in range(n_trials):
        # Splitting data and training RF classifer
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=X.shape[0]-n_tiles)

        rf = RandomForestClassifier(n_estimators=1000)
        rf.fit(X_tr, y_tr)
        accs_rf[i] = rf.score(X_te, y_te)

        # log = LogisticRegression(max_iter=1000)
        # log.fit(X_tr, y_tr)
        # accs_log[i] = log.score(X_te, y_te)

        # nn = MLPClassifier()
        # nn.fit(X_tr, y_tr)
        # accs_nn[i] = nn.score(X_te, y_te)

    print('RF Mean accuracy: {:0.4f}'.format(accs_rf.mean()))
    print('RF Standard deviation: {:0.4f}'.format(accs_rf.std()))
    print('LOG Mean accuracy: {:0.4f}'.format(accs_log.mean()))
    print('LOG Standard deviation: {:0.4f}'.format(accs_log.std()))
    print('nn Mean accuracy: {:0.4f}'.format(accs_nn.mean()))
    print('nn Standard deviation: {:0.4f}'.format(accs_nn.std()))
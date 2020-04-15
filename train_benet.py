import sys
import os
import torch
import argparse
from torch import optim
from time import time

tile2vec_dir = '/raid/users/ebarnett/tile2vec/'
sys.path.append('../')
sys.path.append(tile2vec_dir)

from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from src.tilenet import make_tilenet
from src.training import prep_triplets, train_triplet_epoch


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tile_dir', default='', type=str, help='Path to bigearthnet triplets')
    parser.add_argument('--model_dir', default='', type=str, help='Path to model saving')
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

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cuda = torch.cuda.is_available()

    # Change these arguments to match your directory and desired parameters
    dataloader = triplet_dataloader(img_type    = 'landsat',
                                    tile_dir    = config.tile_dir,
                                    bands       = config.in_channels,
                                    augment     = True,
                                    batch_size  = config.batch_size,
                                    shuffle     = config.shuffle,
                                    num_workers = 4,
                                    n_triplets  = config.n_triplets,
                                    pairs_only  = True)
    print('Dataloader set up complete.')

    # Config the model
    TileNet = make_tilenet(in_channels=config.in_channels, z_dim=config.z_dim)
    TileNet.train()
    if cuda: TileNet.cuda()
    print('TileNet set up complete.')
    optimizer = optim.Adam(TileNet.parameters(), lr=config.lr, betas=(0.5, 0.999))

    print_every = 1000
    save_models = True

    if not os.path.exists(config.model_dir): os.makedirs(config.model_dir)

    t0 = time()
    print('Begin training.................')
    for epoch in range(0, config.epochs):
        (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = \
            train_triplet_epoch(model       = TileNet,
                                cuda        = cuda,
                                dataloader  = dataloader,
                                optimizer   = optimizer,
                                epoch       = epoch+1,
                                margin      = config.margin,
                                l2          = config.l2,
                                print_every = print_every,
                                t0          = t0)

    # Save model after last epoch
    if save_models:
        model_fn = os.path.join(config.model_dir, f'TileNet_epoch{config.epochs}.ckpt')
        torch.save(TileNet.state_dict(), model_fn)
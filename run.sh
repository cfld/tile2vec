#!/bin/bash

# run.sh

PROJECT_ROOT=$(pwd)

# --
# Setup environment
#conda create -y -n tile2vec_env python=3.6
conda init tile2vec_env

#conda install -y -c pytorch pytorch torchvision cudatoolkit=10.0
#conda install -y -c conda-forge scikit-image
#conda install -y -c conda-forge scikit-learn
#
#pip install tqdm
#pip install pandas


# Generate triplets
#python create_triplet.py  --in_dir '/raid/users/bjohnson/data/bigearthnet/bigearthnet/' \
#                          --save_dir '/raid/users/ebarnett/bigearthnet_triplet/'        \
#                          --im_size 50 --neighborhood 100 --n_samples 20000 --n_mosaic 5

# Train tile2vec
#python train_benet.py     --tile_dir '/raid/users/ebarnett/bigearthnet_triplet/' \
#                          --model_dir '/raid/users/ebarnett/tile2vec/models/' \
#                          --batch_size 10 --n_triplets 20000 --z_dim 512 --epochs 5 --lr 0.01 --margin 10 --l2 0.01

## Get labels
#python prep_labels.py

# Generate Features
python featurize_benet.py --tile_dir '/raid/users/bjohnson/data/bigearthnet/bigearthnet/' \
                          --labels_dir '/raid/users/ebarnett/bigearthnet/labels_ben_19.pkl' \
                          --model_dir '/raid/users/ebarnett/tile2vec/models/TileNet_epoch5.ckpt' \
                          --feats_dir '/raid/users/ebarnett/bigearthnet/features/' \
                          --batch_size 16
# Train / Eval Classifier
python train_classifier.py --feats_dir '/raid/users/ebarnett/bigearthnet/features/' \
                           --feats_model 'TileNet_epoch5.ckpt'



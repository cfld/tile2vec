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
#python create_triplet.py  --in_dir "/raid/users/bjohnson/data/bigearthnet/bigearthnet/" \
#                              --save_dir "/raid/users/ebarnett/bigearthnet_triplet/"        \
#                              --im_size 50 --neighborhood 100 --n_samples 2000 --n_mosaic 150

for EPOCH in 5 10
do

    # Train tile2vec
    python train_benet.py     --tile_dir "/raid/users/ebarnett/bigearthnet_triplet/" \
                              --model_dir "/raid/users/ebarnett/tile2vec/models/" \
                              --batch_size 32 --n_triplets 200000 --z_dim 512 --epochs $EPOCH --lr 0.01 --margin 10 --l2 0.01

    # Generate Features
#    python featurize_benet.py --tile_dir "/raid/users/bjohnson/data/bigearthnet/bigearthnet/" \
#                              --labels_dir "/raid/users/ebarnett/bigearthnet/labels_ben_19.pkl" \
#                              --model_dir "/raid/users/ebarnett/tile2vec/models/TileNet_epoch${EPOCH}.ckpt" \
#                              --feats_dir "/raid/users/ebarnett/bigearthnet/features/" \
#                              --batch_size 32
    python embed_eval_benet.py --tile_dir "/raid/users/bjohnson/data/bigearthnet/bigearthnet/" \
                               --labels_dir "/raid/users/ebarnett/bigearthnet/labels_ben_19.pkl" \
                               --model_dir "/raid/users/ebarnett/tile2vec/models/TileNet_epoch${EPOCH}.ckpt" \
                               --feats_dir "/raid/users/ebarnett/bigearthnet/features/" \
                               --batch_size 32


done

# Get labels
python prep_labels.py

for EPOCH in 5 10 20 50
do
    # Train / Eval Classifier
    python embed_eval_benet.py --feats_dir "/raid/users/ebarnett/bigearthnet/features/" \
                               --feats_model "TileNet_epoch${EPOCH}.ckpt"


done
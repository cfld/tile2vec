#!/bin/bash

# featurize.sh

# --
# Featurize

mkdir -p data/naip/feats/states

python tile2vec_featurize.py \
    --indir         /home/bjohnson/projects/moco/data/naip/states/nc \
    --label-channel 4                                                \
    --outdir        data/naip/feats/states/nc > data/naip/feats/states/meta_nc.jl

# # --
# # Downstream models # see moco repo

# python downstream.py \
#     --meta-path  data/naip/feats/states/meta_ca.jl \
#     --feat-path  data/naip/feats/states/ca         \
#     --n-train    1000
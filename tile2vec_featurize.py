#!/usr/bin/env python

"""
    tile2vec_featurize.py
"""

import os
import sys
import json
import bcolz
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.resnet import ResNet18

sys.path.append('/home/bjohnson/projects/moco')
from moco.loader import NAIPValidation

# --
# Helpers

def to_numpy(x):
    return x.detach().cpu().numpy()

def load_tile2vec(weight_path):
    model = ResNet18(tile_size=256)
    model.load_state_dict(torch.load(weight_path))
    model = model.eval()
    model = model.cuda()
    return model

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',         type=str, default='models/naip_trained.ckpt')
    parser.add_argument('--indir',         type=str, default='/home/bjohnson/projects/moco/data/naip/houston')
    parser.add_argument('--outdir',        type=str, default='data/feats/houston')
    parser.add_argument('--label-channel', type=int)
    return parser.parse_args()

args = parse_args()

model = load_tile2vec(args.model)

dataset = NAIPValidation(
    glob_exprs=[os.path.join(args.indir, '*npy')],
    label_channel=args.label_channel,
    normalize=False
)
print(f'n_images={len(dataset)}', file=sys.stderr)

dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=40, pin_memory=True)

all_feats  = bcolz.zeros((len(dataset) * 8, 2048), dtype=np.float32, rootdir=args.outdir, mode='w')

feat_offset = 0
img_offset  = 0
for batch_idx, (xx, yy) in tqdm(enumerate(dataloader), total=len(dataloader)):
    bs, n_patch, n_channel, n_row, n_col = xx.shape
    
    xx = xx.reshape(bs * n_patch, n_channel, n_row, n_col)
    
    if args.label_channel is not None:
        yy   = yy.reshape(bs * n_patch, n_row * n_col)
        labs = yy.mode(axis=-1).values.reshape(bs, n_patch)
    
    with torch.no_grad():
        feats = to_numpy(model.encode(xx.cuda()))
    
    all_feats[feat_offset:feat_offset + feats.shape[0]] = feats
    
    for img_idx in range(bs):
        for patch_idx in range(n_patch):
            print(json.dumps({
                "img_idx"   : img_offset + img_idx,
                "img_name"  : dataset.patch_names[img_offset + img_idx],
                "patch_idx" : patch_idx,
                "lab"       : int(labs[img_idx,patch_idx]) if args.label_channel is not None else None,
            }))
    
    img_offset  += bs
    feat_offset += feats.shape[0]
    
    if batch_idx % 10 == 0:
        all_feats.flush()

all_feats.flush()
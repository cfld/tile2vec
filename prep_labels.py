# !/usr/bin/env python

"""
    prep_labels.py

    !! Seems to be correct, compared to third_party scripts
"""

import os
import re
import json
import pickle
from glob import glob
from tqdm import tqdm

# --
# Load lookups

label_indices = json.load(open('/home/bjohnson/projects/benet/third_party/bigearthnet-19-models/label_indices.json'))
label_conversion = label_indices['label_conversion']

new2idx = label_indices['BigEarthNet-19_labels']
old2idx = label_indices['original_labels']

idx2new = {v: k for k, v in new2idx.items()}

old2new = {}
for new, olds in enumerate(label_conversion):
    for old in olds:
        old2new[old] = new


# --
# Helpers

def convert_labels(patch_dir):
    patch_name = os.path.basename(patch_dir)
    meta_path = os.path.join(patch_dir, patch_name + '_labels_metadata.json')

    meta = json.load(open(meta_path))
    labs = meta['labels']
    old_idxs = [old2idx[l] for l in labs]
    new_idxs = [old2new[i] for i in old_idxs if i in old2new]

    return sorted(list(set(new_idxs)))


# --
# Run

indir = '/raid/users/bjohnson/data/bigearthnet/bigearthnet/'
outpath = '/raid/users/ebarnett/bigearthnet/'

patch_dirs = sorted(glob(os.path.join(indir, '*')))

patch2labs = {}
for patch_dir in tqdm(patch_dirs):
    patch_name = os.path.basename(patch_dir)
    patch2labs[patch_name] = convert_labels(patch_dir)
if not os.path.isdir(outpath): os.mkdir(outpath)
pickle.dump(patch2labs, open(os.path.join(outpath, 'labels_ben_19.pkl'), 'wb'))
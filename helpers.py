import os
import numpy as np
import json
from skimage import io
from skimage.transform import rescale

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

def resize_bands(img, size=120):
    return np.array(rescale(img, size/img.shape[0], anti_aliasing=False))

def load_patch(patch_dir):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    patch_name = os.path.basename(patch_dir)
    patch = [io.imread(os.path.join(patch_dir, f'{patch_name}_{band}.tif')) for band in bands]
    patch = np.stack([resize_bands(xx) for xx in patch], axis=2)
    return patch


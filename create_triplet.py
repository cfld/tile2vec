import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import json
from skimage import io
from skimage.transform import rescale

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

def resize_bands(img, size=120):
    return np.array(rescale(img, size/img.shape[0], anti_aliasing=False))

def load_patch(patch_dir):
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
    patch_name = os.path.basename(patch_dir)
    patch = [io.imread(os.path.join(patch_dir, f'{patch_name}_{band}.tif')) for band in bands]
    patch = np.stack([resize_bands(xx) for xx in patch], axis=2)
    return patch

def get_mosaic(sub):
    px = 120
    tile_source = sub.tile_source.iloc[0]
    sub    = sub.sort_values(['row', 'col']).reset_index(drop=True)
    nrow   = sub.row.max() + 1
    ncol   = sub.col.max() + 1
    mosaic = np.zeros((px * nrow, px * ncol, 12), dtype=np.float)
    for i, row in sub.iterrows():
        start_row = px * row.row
        end_row   = px * row.row + px
        start_col = px * row.col
        end_col   = px * row.col + px
        mosaic[start_row:end_row, start_col:end_col, :] = load_patch(row.patch_dir)

    return mosaic

# Generate triplets functions
def get_centroids(mosaic, n_samples, neighborhood, im_size):

    pad = int(np.ceil(im_size/2) + neighborhood + np.ceil(im_size/2))
    w, h, c = mosaic.shape # c x w x h

    # get anchors
    x_an = np.random.randint(low=pad, high=w-pad, size=n_samples)
    y_an = np.random.randint(low=pad, high=h-pad, size=n_samples)

    # get neighbors
    x_nb = x_an + np.random.randint(low=-np.floor(neighborhood / 2), high=np.floor(neighborhood / 2), size=n_samples)
    y_nb = y_an + np.random.randint(low=-np.floor(neighborhood / 2), high=np.floor(neighborhood / 2), size=n_samples)

    # get distants
    x_di = np.random.choice(w-pad, n_samples, replace=True)
    y_di = np.random.choice(h-pad, n_samples, replace=True)
    bad_idx = np.array([1])

    while bad_idx.shape[0] > 0:
        x_di[bad_idx] = np.random.randint(low=pad, high=w-pad, size=bad_idx.shape[0])
        y_di[bad_idx] = np.random.randint(low=pad, high=h-pad, size=bad_idx.shape[0])
        hyp = np.sqrt(np.abs(x_di - x_an)**2 + np.abs(y_di - y_an)**2)
        bad_idx = np.where(hyp <= neighborhood*2)[0]

    x = np.concatenate((x_an.reshape(-1,1), y_an.reshape(-1,1), x_nb.reshape(-1,1), y_nb.reshape(-1,1), x_di.reshape(-1,1), y_di.reshape(-1,1)), axis = 1)
    df = pd.DataFrame(x, columns = ['x_an', 'y_an', 'x_nb', 'y_nb', 'x_di', 'y_di'])

    return df

def get_subimg(mosaic, x, y, im_size):
    s = int(np.floor(im_size/2))
    return(mosaic[x-s:x+s, y-s:y+s, :])

def save_img(mosaic, idx, row, im_size, save_dir):
    a = get_subimg(mosaic, row.x_an, row.y_an, im_size)
    n = get_subimg(mosaic, row.x_nb, row.y_nb, im_size)
    d = get_subimg(mosaic, row.x_nb, row.y_nb, im_size)

    np.save(os.path.join(save_dir, str(idx) + 'anchor.npy'), a)
    np.save(os.path.join(save_dir, str(idx) + 'neighbor.npy'), n)
    np.save(os.path.join(save_dir, str(idx) + 'distant.npy'), d)


if __name__ == '__main__':

    _ = np.random.seed(3)
    im_size = 50
    neighborhood = 100
    n_samples = 20000

    in_dir = '/raid/users/bjohnson/data/bigearthnet/bigearthnet/'
    save_dir = '/raid/users/ebarnett/bigearthnet_triplet/'
    os.makedirs(save_dir, exist_ok=True)

    # Get patch info
    patch_dirs = glob(os.path.join(in_dir, '*'))
    metas = pd.DataFrame([load_meta(patch_dir) for patch_dir in tqdm(patch_dirs)])

    # Construct mosaic
    tile_sources = metas.tile_source.value_counts()
    tile_sources = tile_sources[tile_sources > 1000] # Skip tiles w/o make patches

    idx = 0
    for t in range(5):
        ts     = tile_sources.index[t]
        meta1  = metas[metas.tile_source == ts]
        meta1  = meta1.sort_values(by=['row', 'col']).reset_index(drop=True)
        mosaic = get_mosaic(meta1)

        # Get centroids
        df_mosaic = get_centroids(mosaic        = mosaic,
                                  n_samples     = n_samples,
                                  neighborhood  = neighborhood,
                                  im_size       = im_size)

        # Pull / Save tiles
        jobs = [delayed(save_img)(mosaic, i+idx, row, im_size, save_dir) for i, row in df_mosaic.iterrows()]
        print("STARTING PARALLEL")
        _ = Parallel(n_jobs=60, backend='multiprocessing', verbose=10)(jobs)

        idx += len(meta1)






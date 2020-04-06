import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')
from helpers import load_meta, resize_bands, load_patch

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', default='', type=str, help='Path to bigearthnet')
    parser.add_argument('--save_dir', default='', type=str, help='Path to save triplets')
    parser.add_argument('--im_size', default=50, type=int, help='size of triplet image')
    parser.add_argument('--neighborhood', default=100, type=int, help='size of neighborhood for tile2vec')
    parser.add_argument('--n_samples', default=20000, type=int, help='number of samples per mosaic')
    parser.add_argument('--n_mosaic', default=5, type=int)
    parser.add_argument('--seed', default=3, type=int)
    config = parser.parse_args()
    _ = np.random.seed(config.seed)

    os.makedirs(config.save_dir, exist_ok=True)

    # Get patch info
    patch_dirs = glob(os.path.join(config.in_dir, '*'))
    metas = pd.DataFrame([load_meta(patch_dir) for patch_dir in tqdm(patch_dirs)])

    # Construct mosaic
    tile_sources = metas.tile_source.value_counts()
    tile_sources = tile_sources[tile_sources > 1000] # Skip tiles w/o make patches

    idx = 0
    print("Getting triplet")
    for t in tqdm(np.arange(0,config.n_mosaic)):
        ts     = tile_sources.index[t]
        meta1  = metas[metas.tile_source == ts]
        meta1  = meta1.sort_values(by=['row', 'col']).reset_index(drop=True)

        mosaic = get_mosaic(meta1)

        # Get centroids
        df_mosaic = get_centroids(mosaic        = mosaic,
                                  n_samples     = config.n_samples,
                                  neighborhood  = config.neighborhood,
                                  im_size       = config.im_size)

        # Pull / Save tiles
        jobs = [delayed(save_img)(mosaic, i+idx, row, config.im_size, config.save_dir) for i, row in df_mosaic.iterrows()]
        print("STARTING PARALLEL")
        _ = Parallel(n_jobs=60, backend='multiprocessing', verbose=0)(jobs)
        idx += len(df_mosaic)






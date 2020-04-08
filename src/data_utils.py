import os
#import gdal
import numpy as np
import imageio
from time import time
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# --
# BigEarthNet

BEN_BAND_STATS = {
    'mean': np.array([
        340.76769064,
        429.9430203,
        614.21682446,
        590.23569706,
        950.68368468,
        1792.46290469,
        2075.46795189,
        2218.94553375,
        2266.46036911,
        2246.0605464,
        1594.42694882,
        1009.32729131
    ])[None,None],
    'std': np.array([
        554.81258967,
        572.41639287,
        582.87945694,
        675.88746967,
        729.89827633,
        1096.01480586,
        1273.45393088,
        1365.45589904,
        1356.13789355,
        1302.3292881,
        1079.19066363,
        818.86747235,
    ])[None,None]
}

def _normalize(x):
    x = np.stack([x[i,:,:] - BEN_BAND_STATS['mean'][0][0,i] / BEN_BAND_STATS['std'][0][0,i] for i in range(12)], axis=0)
    # for i in range(x.shape[2]):
    #     x[:,:,i] = (x[:,:,i] - BEN_BAND_STATS['mean'][i] /BEN_BAND_STATS['std'][i])
    #return (x - BEN_BAND_STATS['mean']) / BEN_BAND_STATS['std']
    return x
def clip_and_scale_image(img, img_type='naip', clip_min=0, clip_max=10000):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """
    if img_type in ['naip', 'rgb']:
        return img / 255
    elif img_type == 'landsat':
        #return np.clip(img, clip_min, clip_max) / (clip_max - clip_min)
        img = _normalize(img)
        return img



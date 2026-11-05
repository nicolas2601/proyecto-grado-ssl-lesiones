"""Classical handcrafted features (LBP, GLCM, colour) for the non-deep baseline; CPU only."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

LBP_BINS = 10
GLCM_PROPS = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation')
COLOR_BINS = 8
FEATURE_DIM = LBP_BINS + len(GLCM_PROPS) * 4 + COLOR_BINS * 3 + 6


def handcrafted_vector(rgb):
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError('Expected an RGB image')
    gray8 = (rgb2gray(rgb) * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray8, 8, 1, 'uniform')
    lbp_hist = np.bincount(lbp.astype(int).ravel(), minlength=LBP_BINS)[:LBP_BINS] / lbp.size
    glcm = graycomatrix(gray8, distances=[1, 2], angles=[0, np.pi / 2], levels=256,
                        symmetric=True, normed=True)
    glcm_feats = np.concatenate([graycoprops(glcm, prop).ravel() for prop in GLCM_PROPS])
    colour_hist = np.concatenate([np.histogram(rgb[..., c], bins=COLOR_BINS, range=(0, 256))[0]
                                  / rgb[..., c].size for c in range(3)])
    moments = np.concatenate([rgb.reshape(-1, 3).mean(0), rgb.reshape(-1, 3).std(0)]) / 255
    vector = np.concatenate([lbp_hist, glcm_feats, colour_hist, moments]).astype(np.float64)
    assert vector.shape == (FEATURE_DIM,)
    return vector


def extract(manifest, output):
    rows = [json.loads(line) for line in Path(manifest).read_text().splitlines()]
    features = np.stack([handcrafted_vector(np.asarray(Image.open(r['image_path']).convert('RGB').resize((224, 224)),
                                                       dtype=np.uint8)) for r in rows])
    np.savez(output, features=features, image_ids=np.asarray([r['image_id'] for r in rows]),
             groups=np.asarray([r['identity_component_id'] for r in rows]))
    return features.shape


if __name__ == '__main__':
    print(extract(sys.argv[1], sys.argv[2]))

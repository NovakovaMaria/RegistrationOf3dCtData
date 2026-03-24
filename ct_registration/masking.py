"""
Specimen masking: Otsu threshold + morphological cleanup.

Produces a binary mask isolating the sandstone specimen from the background.
"""

import numpy as np
from typing import cast
from scipy import ndimage
from skimage.filters import threshold_otsu


def create_specimen_mask(volume: np.ndarray) -> np.ndarray:
    """
    Build a binary specimen mask from a normalised [0, 1] volume.

    Steps:
    1. Otsu threshold
    2. Binary fill holes (per-slice)
    3. Keep largest connected component

    Parameters
    ----------
    volume : ndarray, float [0, 1]

    Returns
    -------
    mask : ndarray bool – specimen mask
    """
    # 1. Otsu threshold
    thresh = threshold_otsu(volume)
    mask = volume > thresh
    print(f"  Otsu threshold = {thresh:.4f}, "
            f"specimen voxels = {mask.sum():,} / {mask.size:,} "
          f"({100 * mask.sum() / mask.size:.1f}%)")

    # 2. Fill holes per slice
    for z in range(mask.shape[0]):
        mask[z] = ndimage.binary_fill_holes(mask[z])

    # 3. Keep largest connected component
    labelled, n_labels = cast(tuple[np.ndarray, int], ndimage.label(mask))
    if n_labels > 1:
        sizes = ndimage.sum(mask, labelled, range(1, n_labels + 1))
        largest = np.argmax(sizes) + 1
        mask = labelled == largest
        print(f"  Kept largest component (of {n_labels}), "
                f"voxels = {mask.sum():,}")

    return mask.astype(bool)

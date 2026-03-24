"""
Specimen masking: Otsu threshold to isolate the sandstone specimen.
"""

import numpy as np
from skimage.filters import threshold_otsu


def create_specimen_mask(volume: np.ndarray) -> np.ndarray:
    """
    Build a binary specimen mask from a normalised [0, 1] volume.

    Parameters
    ----------
    volume : ndarray, float [0, 1]

    Returns
    -------
    mask : ndarray bool – specimen mask
    """
    thresh = threshold_otsu(volume)
    mask = volume > thresh
    print(f"  Otsu threshold = {thresh:.4f}, "
          f"specimen voxels = {mask.sum():,} / {mask.size:,} "
          f"({100 * mask.sum() / mask.size:.1f}%)")

    return mask.astype(bool)

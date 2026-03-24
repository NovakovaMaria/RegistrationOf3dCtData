"""
Specimen masking: Otsu threshold + morphological cleanup.

Produces a binary mask isolating the sandstone specimen from the background,
and an eroded version that avoids edge/interpolation artefacts.
"""

import numpy as np
from scipy import ndimage


def create_specimen_mask(volume: np.ndarray,
                         erode_radius: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a binary specimen mask from a normalised [0, 1] volume.

    Steps:
      1. Otsu threshold
      2. Binary fill holes (per-slice)
      3. Keep largest connected component
      4. Morphological closing to seal small gaps
      5. Erode for the interior-only mask

    Parameters
    ----------
    volume       : ndarray, float [0, 1]
    erode_radius : int – structuring element radius for erosion (voxels)

    Returns
    -------
    mask        : ndarray bool – full specimen mask
    mask_eroded : ndarray bool – eroded mask (interior only)
    """
    # 1. Otsu threshold (custom implementation)
    hist, bin_edges = np.histogram(volume.ravel(), bins=256, range=(0, 1))
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total = hist.sum()
    cum_sum  = np.cumsum(hist)
    cum_mean = np.cumsum(hist * bin_centres)

    best_thresh, best_var = 0.0, 0.0
    for i in range(1, 255):
        w0 = cum_sum[i]
        w1 = total - w0
        if w0 == 0 or w1 == 0:
            continue
        mu0 = cum_mean[i] / w0
        mu1 = (cum_mean[-1] - cum_mean[i]) / w1
        between_var = w0 * w1 * (mu0 - mu1) ** 2
        if between_var > best_var:
            best_var   = between_var
            best_thresh = bin_centres[i]

    thresh = best_thresh
    mask = volume > thresh
    print(f"  Otsu threshold = {thresh:.4f}, "
          f"specimen voxels = {mask.sum():,} / {mask.size:,} "
          f"({100 * mask.sum() / mask.size:.1f}%)")

    # 2. Fill holes per slice
    for z in range(mask.shape[0]):
        mask[z] = ndimage.binary_fill_holes(mask[z])

    # 3. Keep largest connected component
    labelled, n_labels = ndimage.label(mask) # type: ignore
    if n_labels > 1:
        sizes = ndimage.sum(mask, labelled, range(1, n_labels + 1))
        largest = np.argmax(sizes) + 1
        mask = labelled == largest
        print(f"  Kept largest component (of {n_labels}), "
              f"voxels = {mask.sum():,}")

    # 4. Morphological closing
    struct = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_closing(mask, structure=struct, iterations=2)

    # 5. Eroded mask
    erode_struct = np.ones((2 * erode_radius + 1,) * 3, dtype=bool)
    mask_eroded = ndimage.binary_erosion(mask, structure=erode_struct)
    print(f"  Eroded mask ({erode_radius}px): "
          f"{mask_eroded.sum():,} voxels "
          f"({100 * mask_eroded.sum() / mask.sum():.1f}% of specimen)")

    return mask.astype(bool), mask_eroded.astype(bool)
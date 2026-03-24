"""
Specimen masking: Otsu threshold + morphological cleanup.

Produces a binary mask isolating the sandstone specimen from the background,
and an eroded version that avoids edge/interpolation artefacts.
"""

from typing import cast

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu


def create_specimen_mask(
      volume: np.ndarray,
      erode_radius: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
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
      erode_radius : int - structuring element radius for erosion (voxels)

      Returns
      -------
      mask        : ndarray bool - full specimen mask
      mask_eroded : ndarray bool - eroded mask (interior only)
      """
      # 1. Otsu threshold
      thresh = threshold_otsu(volume)
      mask = volume > thresh
      print(
            f"  Otsu threshold = {thresh:.4f}, "
            f"specimen voxels = {mask.sum():,} / {mask.size:,} "
            f"({100 * mask.sum() / mask.size:.1f}%)"
      )

      # 2. Fill holes per slice
      for z in range(mask.shape[0]):
            mask[z] = ndimage.binary_fill_holes(mask[z])

      # 3. Keep largest connected component
      labelled, n_labels = cast(tuple[np.ndarray, int], ndimage.label(mask))
      if n_labels > 1:
            sizes = ndimage.sum(mask, labelled, range(1, n_labels + 1))
            largest = np.argmax(sizes) + 1
            mask = labelled == largest
            print(f"  Kept largest component (of {n_labels}), voxels = {mask.sum():,}")

      # 4. Morphological closing
      struct = ndimage.generate_binary_structure(3, 1)
      mask = ndimage.binary_closing(mask, structure=struct, iterations=2)

      # 5. Eroded mask
      erode_struct = np.ones((2 * erode_radius + 1,) * 3, dtype=bool)
      mask_eroded = ndimage.binary_erosion(mask, structure=erode_struct)
      print(
            f"  Eroded mask ({erode_radius}px): "
            f"{mask_eroded.sum():,} voxels "
            f"({100 * mask_eroded.sum() / mask.sum():.1f}% of specimen)"
      )

      return mask.astype(bool), mask_eroded.astype(bool)
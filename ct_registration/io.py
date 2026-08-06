"""
Data loading and saving utilities for TIF volumes.
"""

import os
import numpy as np
import tifffile
import SimpleITK as sitk

from .config import FIXED_PATH, MOVING_PATH, REGISTERED_PATH, RESULTS_DIR


def load_tif_as_sitk(path: str) -> sitk.Image:
    """Load a multi-page TIF stack and return a SimpleITK 3-D image (float32)."""
    print(f"  Loading {os.path.basename(path)} ...")
    arr = tifffile.imread(path)                       # (Z, Y, X)
    print(f"    Shape: {arr.shape}, dtype: {arr.dtype}, "
          f"range: [{arr.min()}, {arr.max()}]")
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing((1.0, 1.0, 1.0))
    img.SetOrigin((0.0, 0.0, 0.0))
    return img


def load_volumes_sitk():
    """Load the fixed and moving volumes as SimpleITK images."""
    print("\n── Loading Data ──")
    fixed = load_tif_as_sitk(FIXED_PATH)
    moving = load_tif_as_sitk(MOVING_PATH)
    return fixed, moving


def sitk_to_numpy(img: sitk.Image) -> np.ndarray:
    """Convert a SimpleITK image to a NumPy array."""
    return sitk.GetArrayFromImage(img)


def save_registered_volume(arr: np.ndarray) -> str:
    """Save the registered volume as a float32 TIF stack; return file path."""
    out_path = os.path.join(RESULTS_DIR, "VEC4-02-b2_registered.tif")
    tifffile.imwrite(out_path, arr.astype(np.float32))
    print(f"\n  Saved registered volume → {out_path}")
    return out_path


def load_all_numpy():
    """
    Load fixed, moving, and registered volumes as normalised [0, 1] NumPy arrays.
    Used by the visualisation pipeline (requires register_ct.py to have run first).
    """
    print("Loading volumes ...")
    fixed = tifffile.imread(FIXED_PATH).astype(np.float64)
    moving = tifffile.imread(MOVING_PATH).astype(np.float64)
    registered = tifffile.imread(REGISTERED_PATH).astype(np.float64)

    gmax = max(fixed.max(), moving.max(), registered.max())
    fixed /= gmax
    moving /= gmax
    registered /= gmax

    print(f"  Fixed:      {fixed.shape}")
    print(f"  Moving:     {moving.shape}")
    print(f"  Registered: {registered.shape}")
    return fixed, moving, registered

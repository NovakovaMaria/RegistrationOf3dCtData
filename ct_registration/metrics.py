"""
Quantitative comparison metrics for registration quality assessment.

Supports whole-volume and masked (specimen-only / eroded) evaluation.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_global_metrics(a: np.ndarray, b: np.ndarray):
    """Compute MSE and NCC between two normalised [0, 1] arrays."""
    mse = float(np.mean((a - b) ** 2))
    az = a - a.mean()
    bz = b - b.mean()
    ncc = float(np.sum(az * bz) / (np.sqrt(np.sum(az**2) * np.sum(bz**2)) + 1e-12))
    return mse, ncc


def compute_masked_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    """MSE and NCC evaluated only on voxels where mask is True."""
    av, bv = a[mask], b[mask]
    mse = float(np.mean((av - bv) ** 2))
    az = av - av.mean()
    bz = bv - bv.mean()
    ncc = float(np.sum(az * bz) / (np.sqrt(np.sum(az**2) * np.sum(bz**2)) + 1e-12))
    return mse, ncc


def compute_ssim_per_slice(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM across all axial slices."""
    vals = [ssim(a[z], b[z], data_range=1.0) for z in range(a.shape[0])]
    return float(np.mean(vals))


def quantitative_comparison(fixed: np.ndarray, moving: np.ndarray,
                            registered: np.ndarray,
                            mask: np.ndarray | None = None,
                            mask_eroded: np.ndarray | None = None) -> dict:
    """
    Compute and print metrics before and after registration.

    Arrays are normalised internally to [0, 1].

    Returns
    -------
    dict – {label: {metric_name: value}}
    """
    print("\n── Quantitative Comparison ──")
    results = {}

    for label, comp in [
        ("Before registration", moving),
        ("After registration",  registered),
    ]:
        fmax = max(fixed.max(), comp.max())
        f_n = fixed.astype(np.float64) / fmax
        c_n = comp.astype(np.float64)  / fmax

        mse, ncc = compute_global_metrics(f_n, c_n)
        mean_ssim = compute_ssim_per_slice(f_n, c_n)

        entry = {"MSE": mse, "NCC": ncc, "SSIM": mean_ssim}

        if mask is not None:
            mse_m, ncc_m = compute_masked_metrics(f_n, c_n, mask)
            entry["MSE_mask"] = mse_m
            entry["NCC_mask"] = ncc_m

        if mask_eroded is not None:
            mse_e, ncc_e = compute_masked_metrics(f_n, c_n, mask_eroded)
            entry["MSE_eroded"] = mse_e
            entry["NCC_eroded"] = ncc_e

        results[label] = entry
        print(f"\n  {label}:")
        for k, v in entry.items():
            print(f"    {k:20s} = {v:.6f}")

    return results
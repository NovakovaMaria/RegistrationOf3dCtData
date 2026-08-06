"""
Quantitative comparison metrics for registration quality assessment.

Supports both whole-volume and masked (specimen-only / eroded) evaluation.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


# ─── Low-level helpers ───────────────────────────────────────────────────────

def compute_global_metrics(a: np.ndarray, b: np.ndarray):
    """
    Compute MSE and NCC between two arrays (assumed normalised [0, 1]).

    Returns
    -------
    mse : float
    ncc : float
    """
    mse = float(np.mean((a - b) ** 2))
    az = a - a.mean()
    bz = b - b.mean()
    ncc = float(np.sum(az * bz) / (np.sqrt(np.sum(az**2) * np.sum(bz**2)) + 1e-12))
    return mse, ncc


def compute_masked_metrics(a: np.ndarray, b: np.ndarray, mask: np.ndarray):
    """
    MSE and NCC evaluated only on voxels where *mask* is True.

    Returns
    -------
    mse : float
    ncc : float
    """
    av = a[mask]
    bv = b[mask]
    mse = float(np.mean((av - bv) ** 2))
    az = av - av.mean()
    bz = bv - bv.mean()
    ncc = float(np.sum(az * bz) / (np.sqrt(np.sum(az**2) * np.sum(bz**2)) + 1e-12))
    return mse, ncc


def compute_ssim_per_slice(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return an array of SSIM values, one per axial (Z) slice."""
    vals = []
    for z in range(a.shape[0]):
        vals.append(ssim(a[z], b[z], data_range=1.0))
    return np.array(vals)


# ─── Robust difference statistics ────────────────────────────────────────────

def _diff_stats(abs_diff: np.ndarray, tag: str = "") -> dict:
    """Return mean, std, max, and 99.9-th percentile of |diff|."""
    return {
        f"AbsDiff_mean{tag}":  float(abs_diff.mean()),
        f"AbsDiff_std{tag}":   float(abs_diff.std()),
        f"AbsDiff_max{tag}":   float(abs_diff.max()),
        f"AbsDiff_P99.9{tag}": float(np.percentile(abs_diff, 99.9)),
    }


# ─── Public API ──────────────────────────────────────────────────────────────

def quantitative_comparison(fixed_arr: np.ndarray,
                            moving_arr: np.ndarray,
                            registered_arr: np.ndarray,
                            mask: np.ndarray | None = None,
                            mask_eroded: np.ndarray | None = None) -> dict:
    """
    Compute and print quantitative metrics before and after registration.

    All arrays are expected in their original dtype; they are normalised
    internally to [0, 1] for fair comparison.

    If *mask* (and optionally *mask_eroded*) are provided, metrics are
    also reported inside the specimen and inside the eroded interior.

    Returns
    -------
    dict  – nested ``{label: {metric_name: value}}``
    """
    print("\n── Quantitative Comparison ──")
    results = {}

    for label, comp in [
        ("Before registration (moving vs fixed)", moving_arr),
        ("After registration (registered vs fixed)", registered_arr),
    ]:
        fmax = max(fixed_arr.max(), comp.max())
        f_norm = fixed_arr.astype(np.float64) / fmax
        c_norm = comp.astype(np.float64) / fmax

        # ── Whole-volume metrics ──
        mse, ncc = compute_global_metrics(f_norm, c_norm)
        ssim_vals = compute_ssim_per_slice(f_norm, c_norm)
        mean_ssim = float(ssim_vals.mean())

        abs_diff = np.abs(f_norm - c_norm)

        entry: dict = {
            "MSE": mse,
            "NCC": ncc,
            "SSIM": mean_ssim,
            **_diff_stats(abs_diff),
        }

        # ── Masked (specimen-only) metrics ──
        if mask is not None:
            mse_m, ncc_m = compute_masked_metrics(f_norm, c_norm, mask)
            entry["MSE_mask"] = mse_m
            entry["NCC_mask"] = ncc_m
            entry.update(_diff_stats(abs_diff[mask], tag="_mask"))

        # ── Eroded mask (interior-only) metrics ──
        if mask_eroded is not None:
            mse_e, ncc_e = compute_masked_metrics(f_norm, c_norm, mask_eroded)
            entry["MSE_eroded"] = mse_e
            entry["NCC_eroded"] = ncc_e
            entry.update(_diff_stats(abs_diff[mask_eroded], tag="_eroded"))

        results[label] = entry

        print(f"\n  {label}:")
        for k, v in entry.items():
            print(f"    {k:25s} = {v:.6f}")

    return results

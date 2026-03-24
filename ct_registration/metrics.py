"""
Quantitative comparison metrics for registration quality assessment.
"""

import numpy as np


def compute_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Squared Error between two arrays."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def compute_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Cross-Correlation between two arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    az = a - a.mean()
    bz = b - b.mean()
    return float(np.sum(az * bz) / (np.sqrt(np.sum(az**2) * np.sum(bz**2)) + 1e-12))


def compute_masked_metrics(a: np.ndarray, b: np.ndarray,
                            mask: np.ndarray) -> dict:
    """MSE and NCC evaluated only on voxels where mask is True."""
    return {
        "MSE": compute_mse(a[mask], b[mask]),
        "NCC": compute_ncc(a[mask], b[mask]),
    }
"""
Quick-preview visualisations produced during the registration pipeline.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import RESULTS_DIR


def plot_central_slices(fixed, moving, registered):
    """Central slice for each view direction: fixed, moving, registered + diff."""
    print("\n── Generating slice comparison figures ──")

    nz, ny, nx = fixed.shape
    views = {
        "axial":    (nz // 2, lambda v, i: v[i, :, :]),
        "coronal":  (ny // 2, lambda v, i: v[:, i, :]),
        "sagittal": (nx // 2, lambda v, i: v[:, :, i]),
    }

    for view_name, (idx, slicer) in views.items():
        f_sl = slicer(fixed,      idx).astype(np.float64)
        m_sl = slicer(moving,     idx).astype(np.float64)
        r_sl = slicer(registered, idx).astype(np.float64)

        vmin = min(f_sl.min(), m_sl.min(), r_sl.min())
        vmax = max(f_sl.max(), m_sl.max(), r_sl.max())

        diff_before = f_sl - m_sl
        diff_after  = f_sl - r_sl
        dmax = max(np.abs(diff_before).max(), np.abs(diff_after).max())

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Central {view_name} slice (index {idx})", fontsize=14)

        axes[0, 0].imshow(f_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Fixed")
        axes[0, 1].imshow(m_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("Moving")
        axes[0, 2].imshow(r_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 2].set_title("Registered")

        axes[1, 0].imshow(diff_before, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[1, 0].set_title("Difference: Fixed − Moving")
        axes[1, 1].imshow(diff_after, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[1, 1].set_title("Difference: Fixed − Registered")
        axes[1, 2].axis("off")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        fname = f"slices_{view_name}.png"
        fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")
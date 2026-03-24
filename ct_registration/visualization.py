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


def plot_checkerboard(fixed, registered, block_size=32):
    """Checkerboard comparison on the central axial slice."""
    print("\n── Generating checkerboard figure ──")
    z = fixed.shape[0] // 2
    f_sl = fixed[z].astype(np.float64)
    r_sl = registered[z].astype(np.float64)

    h, w = f_sl.shape
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((yy // block_size) + (xx // block_size)) % 2 == 0
    combined = np.where(checker, f_sl, r_sl)

    vmin = min(f_sl.min(), r_sl.min())
    vmax = max(f_sl.max(), r_sl.max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(f_sl, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Fixed")
    axes[1].imshow(r_sl, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Registered")
    axes[2].imshow(combined, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Checkerboard ({block_size}px)")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "checkerboard.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved checkerboard.png")


def plot_misalignment_overlay(fixed, moving, registered):
    """Magenta/green overlay before and after registration."""
    print("\n── Generating misalignment overlay ──")
    z = fixed.shape[0] // 2
    f_sl = fixed[z].astype(np.float64)
    m_sl = moving[z].astype(np.float64)
    r_sl = registered[z].astype(np.float64)

    def _overlay(a, b):
        vmax = max(a.max(), b.max()) + 1e-12
        overlay = np.zeros((*a.shape, 3))
        overlay[..., 0] = a / vmax   # red channel   - fixed
        overlay[..., 1] = b / vmax   # green channel - compared
        overlay[..., 2] = a / vmax   # blue channel  - fixed (magenta = R+B)
        return np.clip(overlay, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(_overlay(f_sl, m_sl))
    axes[0].set_title("Before registration (Fixed=magenta, Moving=green)")
    axes[1].imshow(_overlay(f_sl, r_sl))
    axes[1].set_title("After registration (Fixed=magenta, Registered=green)")
    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "misalignment_overlay.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved misalignment_overlay.png")


def plot_difference_histogram(fixed, moving, registered):
    """Histogram of voxel intensity differences before/after registration."""
    print("\n── Generating difference histogram ──")
    diff_before = (fixed - moving).ravel()
    diff_after  = (fixed - registered).ravel()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-0.5, 0.5, 201).tolist()
    ax.hist(diff_before, bins=bins, alpha=0.5, density=True,
            color="red",  label="Before registration")
    ax.hist(diff_after,  bins=bins, alpha=0.5, density=True,
            color="blue", label="After registration")
    ax.set_xlabel("Normalised intensity difference (Fixed − Compared)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of voxel differences")
    ax.legend()
    ax.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "difference_histogram.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved difference_histogram.png")
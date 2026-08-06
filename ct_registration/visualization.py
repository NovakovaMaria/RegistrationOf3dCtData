"""
Quick-preview visualisations produced during the registration pipeline.

These are simpler plots saved alongside the registration output.
For presentation-quality figures see ``figures.py``.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from .config import RESULTS_DIR


def plot_central_slices(fixed_arr, moving_arr, registered_arr):
    """Central slices along each axis: fixed, moving, registered + diffs."""
    print("\n── Generating slice comparison figures ──")

    nz, ny, nx = fixed_arr.shape
    slices = {
        "axial (Z)":    (nz // 2, lambda a: a[nz // 2, :, :]),
        "coronal (Y)":  (ny // 2, lambda a: a[:, ny // 2, :]),
        "sagittal (X)": (nx // 2, lambda a: a[:, :, nx // 2]),
    }

    for view_name, (idx, slicer) in slices.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"Central {view_name} slice (index {idx})", fontsize=16)

        f_sl = slicer(fixed_arr).astype(np.float64)
        m_sl = slicer(moving_arr).astype(np.float64)
        r_sl = slicer(registered_arr).astype(np.float64)

        vmin = min(f_sl.min(), m_sl.min(), r_sl.min())
        vmax = max(f_sl.max(), m_sl.max(), r_sl.max())

        # Row 1: individual images
        axes[0, 0].imshow(f_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 0].set_title("Fixed (before deformation)")
        axes[0, 1].imshow(m_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 1].set_title("Moving (after deformation)")
        axes[0, 2].imshow(r_sl, cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, 2].set_title("Registered (aligned)")

        # Row 2: difference maps + overlay
        diff_before = f_sl - m_sl
        diff_after = f_sl - r_sl
        dmax = max(np.abs(diff_before).max(), np.abs(diff_after).max())

        axes[1, 0].imshow(diff_before, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[1, 0].set_title("Difference: Fixed − Moving")
        axes[1, 1].imshow(diff_after, cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[1, 1].set_title("Difference: Fixed − Registered")

        overlay = np.zeros((*f_sl.shape, 3), dtype=np.float64)
        f_n = (f_sl - vmin) / (vmax - vmin + 1e-12)
        r_n = (r_sl - vmin) / (vmax - vmin + 1e-12)
        overlay[..., 0] = f_n
        overlay[..., 1] = r_n
        overlay[..., 2] = f_n
        axes[1, 2].imshow(np.clip(overlay, 0, 1))
        axes[1, 2].set_title("Overlay: Fixed (magenta) / Registered (green)")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        fname = f"slices_{view_name.split()[0]}.png"
        fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_checkerboard(fixed_arr, registered_arr, block_size=32):
    """Checkerboard comparison on the central axial slice."""
    print("\n── Generating checkerboard figure ──")
    z_mid = fixed_arr.shape[0] // 2

    f_sl = fixed_arr[z_mid].astype(np.float64)
    r_sl = registered_arr[z_mid].astype(np.float64)

    h, w = f_sl.shape
    yy, xx = np.mgrid[0:h, 0:w]
    checker = ((yy // block_size) + (xx // block_size)) % 2 == 0
    combined = np.where(checker, f_sl, r_sl)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmin = min(f_sl.min(), r_sl.min())
    vmax = max(f_sl.max(), r_sl.max())

    axes[0].imshow(f_sl, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Fixed (before deformation)")
    axes[1].imshow(r_sl, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].set_title("Registered (aligned)")
    axes[2].imshow(combined, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"Checkerboard ({block_size}px blocks)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "checkerboard.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved checkerboard.png")


def plot_misalignment_overlay(fixed_arr, moving_arr, registered_arr):
    """Magenta / green overlay before and after registration."""
    print("\n── Generating misalignment overlay ──")
    z_mid = fixed_arr.shape[0] // 2

    def _make_panel(fig_axes, title, first_arr, second_arr, overlay_label):
        f_sl = first_arr[z_mid].astype(np.float64)
        for ax, (t, comp) in zip(
            fig_axes,
            [(f"Before deformation (Fixed)", first_arr),
             (title, second_arr),
             (overlay_label, None)],
        ):
            if comp is not None:
                c_sl = comp[z_mid].astype(np.float64)
                vmin = min(f_sl.min(), c_sl.min())
                vmax = max(f_sl.max(), c_sl.max())
                ax.imshow((c_sl - vmin) / (vmax - vmin + 1e-12), cmap="gray")
            else:
                s_sl = second_arr[z_mid].astype(np.float64)
                vmin = min(f_sl.min(), s_sl.min())
                vmax = max(f_sl.max(), s_sl.max())
                overlay = np.zeros((*f_sl.shape, 3))
                overlay[..., 0] = (f_sl - vmin) / (vmax - vmin + 1e-12)
                overlay[..., 1] = (s_sl - vmin) / (vmax - vmin + 1e-12)
                overlay[..., 2] = (f_sl - vmin) / (vmax - vmin + 1e-12)
                ax.imshow(np.clip(overlay, 0, 1))
            ax.set_title(t, fontsize=12)
            ax.axis("off")

    # Before registration
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _make_panel(axes, "After deformation (Moving)", fixed_arr, moving_arr,
                "Misalignment: Fixed vs Moving")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "misalignment_before.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # After registration
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _make_panel(axes, "Registered (aligned)", fixed_arr, registered_arr,
                "Residual misalignment: Fixed vs Registered")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "misalignment_after.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved misalignment_before.png, misalignment_after.png")


def plot_difference_histogram(fixed_arr, moving_arr, registered_arr):
    """Histogram of voxel intensity differences before/after registration."""
    print("\n── Generating difference histogram ──")
    fmax = max(fixed_arr.max(), moving_arr.max(), registered_arr.max())
    f = fixed_arr.astype(np.float64) / fmax
    m = moving_arr.astype(np.float64) / fmax
    r = registered_arr.astype(np.float64) / fmax

    diff_before = (f - m).ravel()
    diff_after = (f - r).ravel()

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(-0.5, 0.5, 201)
    ax.hist(diff_before, bins=bins, alpha=0.5, label="Before registration",
            density=True, color="red")
    ax.hist(diff_after, bins=bins, alpha=0.5, label="After registration",
            density=True, color="blue")
    ax.set_xlabel("Normalised intensity difference (Fixed − Compared)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of voxel differences")
    ax.legend()
    ax.set_xlim(-0.5, 0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "difference_histogram.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved difference_histogram.png")

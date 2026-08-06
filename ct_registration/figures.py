"""
Presentation-quality figures (fig1-fig8) for the registration results.

Run *after* the registration pipeline has produced the registered volume.
Each function takes the three normalised [0, 1] volumes and saves one PNG
into ``config.RESULTS_DIR``.
"""

import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.metrics import structural_similarity as ssim

from .config import RESULTS_DIR, TEAL, CORAL, SLATE
from .metrics import compute_global_metrics, compute_ssim_per_slice


# ─── Figure 1 ────────────────────────────────────────────────────────────────

def fig1_overview(fixed, moving, registered):
    """3 × 3 grid: (axial, coronal, sagittal) × (fixed, moving, registered)."""
    print("Generating Fig 1: Overview ...")
    nz, ny, nx = fixed.shape

    views = [
        ("Axial (Z)",    nz // 2, lambda v, i: v[i, :, :]),
        ("Coronal (Y)",  ny // 2, lambda v, i: v[:, i, :]),
        ("Sagittal (X)", nx // 2, lambda v, i: v[:, :, i]),
    ]
    cols = [
        ("Before deformation\n(Fixed)",   fixed),
        ("After deformation\n(Moving)",   moving),
        ("After registration\n(Aligned)", registered),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 14))
    fig.suptitle("3D CT Registration - Volume Overview", fontsize=18,
                 fontweight="bold", color=SLATE, y=0.98)

    for row, (view_name, idx, slicer) in enumerate(views):
        for col, (col_name, vol) in enumerate(cols):
            ax = axes[row, col]
            ax.imshow(slicer(vol, idx), cmap="gray", vmin=0, vmax=0.85,
                      aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_name, fontsize=12, pad=10, color=SLATE)
            if col == 0:
                ax.set_ylabel(view_name, fontsize=12, fontweight="bold",
                              color=TEAL, labelpad=10)
            for spine in ax.spines.values():
                spine.set_color("#cccccc")
                spine.set_linewidth(0.5)

    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(os.path.join(RESULTS_DIR, "fig1_overview.png"))
    plt.close(fig)
    print("  Saved fig1_overview.png")


# ─── Figure 2 ────────────────────────────────────────────────────────────────

def fig2_misalignment(fixed, moving, registered):
    """Magenta/green overlays before vs after with zoomed insets."""
    print("Generating Fig 2: Misalignment overlay ...")
    z_mid = fixed.shape[0] // 2
    vmax = 0.85

    f_sl = fixed[z_mid]
    m_sl = moving[z_mid]
    r_sl = registered[z_mid]

    def _overlay(a, b):
        a_n = np.clip(a / vmax, 0, 1)
        b_n = np.clip(b / vmax, 0, 1)
        return np.clip(np.stack([a_n, b_n, a_n], axis=-1), 0, 1)

    overlay_before = _overlay(f_sl, m_sl)
    overlay_after = _overlay(f_sl, r_sl)

    fig = plt.figure(figsize=(20, 9))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.02], wspace=0.08)

    # --- Before ---
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(overlay_before)
    ax1.set_title("Before Registration", fontsize=15, fontweight="bold",
                  color=CORAL, pad=12)
    ax1.set_xticks([]); ax1.set_yticks([])

    y1, y2, x1, x2 = 30, 120, 150, 250
    axins1 = ax1.inset_axes([0.58, 0.60, 0.40, 0.38])
    axins1.imshow(overlay_before[y1:y2, x1:x2])
    axins1.set_xticks([]); axins1.set_yticks([])
    for s in axins1.spines.values():
        s.set_color("yellow"); s.set_linewidth(2)
    _, lines1 = ax1.indicate_inset_zoom(axins1, edgecolor="yellow", linewidth=1.2)
    for ln in lines1:
        ln.set(linewidth=0.6, linestyle="--", alpha=0.5)

    # --- After ---
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(overlay_after)
    ax2.set_title("After Registration", fontsize=15, fontweight="bold",
                  color=TEAL, pad=12)
    ax2.set_xticks([]); ax2.set_yticks([])

    axins2 = ax2.inset_axes([0.58, 0.60, 0.40, 0.38])
    axins2.imshow(overlay_after[y1:y2, x1:x2])
    axins2.set_xticks([]); axins2.set_yticks([])
    for s in axins2.spines.values():
        s.set_color("yellow"); s.set_linewidth(2)
    _, lines2 = ax2.indicate_inset_zoom(axins2, edgecolor="yellow", linewidth=1.2)
    for ln in lines2:
        ln.set(linewidth=0.6, linestyle="--", alpha=0.5)

    # --- Legend ---
    ax_leg = fig.add_subplot(gs[2])
    ax_leg.axis("off")
    legend_elems = [
        plt.Rectangle((0, 0), 1, 1, fc=(0.8, 0, 0.8), label="Fixed"),
        plt.Rectangle((0, 0), 1, 1, fc=(0, 0.8, 0),   label="Compared"),
        plt.Rectangle((0, 0), 1, 1, fc=(0.7, 0.7, 0.7), label="Aligned"),
    ]
    ax_leg.legend(handles=legend_elems, loc="center", fontsize=10,
                  frameon=True, fancybox=True, shadow=True)

    fig.suptitle("Misalignment Comparison - Axial Slice", fontsize=18,
                 fontweight="bold", color=SLATE, y=1.02)
    fig.savefig(os.path.join(RESULTS_DIR, "fig2_misalignment.png"),
               bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_misalignment.png")


# ─── Figure 3 ────────────────────────────────────────────────────────────────

def fig3_difference_maps(fixed, moving, registered, mask=None):
    """Absolute difference maps + improvement maps for all 3 views.

    When *mask* is supplied the background is zeroed out so only the
    specimen interior contributes to the visual.
    """
    print("Generating Fig 3: Difference maps ...")
    nz, ny, nx = fixed.shape
    views = [
        ("Axial",    nz // 2, lambda v, i: v[i, :, :]),
        ("Coronal",  ny // 2, lambda v, i: v[:, i, :]),
        ("Sagittal", nx // 2, lambda v, i: v[:, :, i]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    title_suffix = " (specimen only)" if mask is not None else ""
    fig.suptitle(f"Difference Maps - Before vs After Registration{title_suffix}",
                 fontsize=18, fontweight="bold", color=SLATE, y=0.98)

    for row, (view_name, idx, slicer) in enumerate(views):
        f_sl = slicer(fixed, idx)
        m_sl = slicer(moving, idx)
        r_sl = slicer(registered, idx)

        diff_before = np.abs(f_sl - m_sl)
        diff_after = np.abs(f_sl - r_sl)

        # zero-out background when mask available
        if mask is not None:
            m_sl_2d = slicer(mask.astype(float), idx) > 0.5
            diff_before = diff_before * m_sl_2d
            diff_after  = diff_after  * m_sl_2d

        dmax = max(diff_before.max(), diff_after.max())

        im0 = axes[row, 0].imshow(diff_before, cmap="inferno", vmin=0, vmax=dmax)
        if row == 0:
            axes[row, 0].set_title("|Fixed − Moving|", fontsize=12, color=CORAL)
        axes[row, 0].set_ylabel(f"{view_name} (slice {idx})",
                                fontsize=11, fontweight="bold", color=TEAL)

        im1 = axes[row, 1].imshow(diff_after, cmap="inferno", vmin=0, vmax=dmax)
        if row == 0:
            axes[row, 1].set_title("|Fixed − Registered|", fontsize=12, color=TEAL)

        improvement = diff_before - diff_after
        imax = max(abs(improvement.min()), abs(improvement.max()))
        im2 = axes[row, 2].imshow(improvement, cmap="RdYlGn", vmin=-imax, vmax=imax)
        if row == 0:
            axes[row, 2].set_title("Improvement\n(green = better)", fontsize=12,
                                   color=SLATE)

        for ax, im in [(axes[row, 0], im0), (axes[row, 1], im1), (axes[row, 2], im2)]:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.05)
            plt.colorbar(im, cax=cax)

        for ax in axes[row, :]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.subplots_adjust(wspace=0.25, hspace=0.15)
    fig.savefig(os.path.join(RESULTS_DIR, "fig3_difference_maps.png"))
    plt.close(fig)
    print("  Saved fig3_difference_maps.png")


# ─── Figure 4 ────────────────────────────────────────────────────────────────

def fig4_checkerboard(fixed, registered, block_size=16):
    """Colour-tinted checkerboard at 5 Z depths + zoomed edge inset.

    Fixed tiles are tinted cyan, registered tiles orange, making the
    alternating pattern unmistakable.  The 6th column zooms into a
    specimen edge/notch region for fine-detail verification.
    """
    print("Generating Fig 4: Checkerboard ...")
    nz, ny, nx = fixed.shape
    z_positions = np.linspace(nz * 0.15, nz * 0.85, 5, dtype=int)

    TINT_FIXED = np.array([0.65, 0.92, 0.95])   # light cyan
    TINT_REG   = np.array([0.98, 0.82, 0.60])   # light orange
    TINT_STRENGTH = 0.25

    def _build_checker(f_sl, r_sl, bs):
        h, w = f_sl.shape
        f_n = np.clip(f_sl / 0.85, 0, 1)
        r_n = np.clip(r_sl / 0.85, 0, 1)
        yy, xx = np.mgrid[0:h, 0:w]
        cmask = ((yy // bs) + (xx // bs)) % 2 == 0
        rgb = np.zeros((h, w, 3))
        for c in range(3):
            rgb[..., c] = np.where(
                cmask,
                f_n * (1 - TINT_STRENGTH) + TINT_STRENGTH * TINT_FIXED[c],
                r_n * (1 - TINT_STRENGTH) + TINT_STRENGTH * TINT_REG[c],
            )
        for y_line in range(bs, h, bs):
            rgb[max(y_line - 1, 0):y_line, :, :] = 0.35
        for x_line in range(bs, w, bs):
            rgb[:, max(x_line - 1, 0):x_line, :] = 0.35
        return np.clip(rgb, 0, 1), f_n, r_n

    fig, axes = plt.subplots(2, 6, figsize=(30, 10))
    fig.suptitle("Checkerboard Comparison - Fixed (cyan) vs Registered (orange)",
                 fontsize=18, fontweight="bold", color=SLATE, y=0.98)

    for col, z in enumerate(z_positions):
        rgb, f_n, r_n = _build_checker(fixed[z], registered[z], block_size)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(f"Z = {z}", fontsize=11, fontweight="bold")
        axes[1, col].imshow(np.clip(np.stack([f_n, r_n, f_n], axis=-1), 0, 1))

    # Column 5: zoomed edge/notch inset (top-right specimen corner)
    z_mid = nz // 2
    y1, y2, x1, x2 = 30, 120, 150, 250  # ROI at specimen edge
    roi_f = fixed[z_mid, y1:y2, x1:x2]
    roi_r = registered[z_mid, y1:y2, x1:x2]
    rgb_zoom, fn_z, rn_z = _build_checker(roi_f, roi_r, block_size)
    axes[0, 5].imshow(rgb_zoom)
    axes[0, 5].set_title(f"Edge zoom\nZ={z_mid}, y=[{y1}:{y2}]", fontsize=10,
                         fontweight="bold", color=CORAL)
    axes[1, 5].imshow(np.clip(np.stack([fn_z, rn_z, fn_z], axis=-1), 0, 1))

    axes[0, 0].set_ylabel("Checkerboard\n(F=cyan, R=orange)", fontsize=11,
                           fontweight="bold", color=TEAL)
    axes[1, 0].set_ylabel("Overlay\n(F=magenta, R=green)", fontsize=11,
                           fontweight="bold", color=TEAL)

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    plt.subplots_adjust(wspace=0.05, hspace=0.08)
    fig.savefig(os.path.join(RESULTS_DIR, "fig4_checkerboard.png"))
    plt.close(fig)
    print("  Saved fig4_checkerboard.png")


# ─── Figure 5 ────────────────────────────────────────────────────────────────

def fig5_metrics(fixed, moving, registered):
    """Bar charts + slice-by-slice SSIM/NCC plots.  Returns metrics dict."""
    print("Generating Fig 5: Quantitative metrics ...")

    mse_before, ncc_before = compute_global_metrics(fixed, moving)
    mse_after,  ncc_after  = compute_global_metrics(fixed, registered)

    ssim_before = compute_ssim_per_slice(fixed, moving)
    ssim_after  = compute_ssim_per_slice(fixed, registered)

    ncc_before_sl = np.array([compute_global_metrics(fixed[z], moving[z])[1]
                              for z in range(fixed.shape[0])])
    ncc_after_sl  = np.array([compute_global_metrics(fixed[z], registered[z])[1]
                              for z in range(fixed.shape[0])])

    mean_ssim_before = float(ssim_before.mean())
    mean_ssim_after  = float(ssim_after.mean())

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2], hspace=0.35, wspace=0.35)
    fig.suptitle("Quantitative Registration Quality",
                 fontsize=18, fontweight="bold", color=SLATE, y=0.98)

    # MSE bar
    ax_mse = fig.add_subplot(gs[0, 0])
    bars = ax_mse.bar(["Before", "After"], [mse_before, mse_after],
                      color=[CORAL, TEAL], width=0.5, alpha=0.85)
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("Mean Squared Error", fontweight="bold")
    for bar, val in zip(bars, [mse_before, mse_after]):
        ax_mse.text(bar.get_x() + bar.get_width() / 2., bar.get_height() * 1.02,
                    f"{val:.6f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
    improvement = (1 - mse_after / mse_before) * 100
    ax_mse.annotate(f"↓ {improvement:.1f}%", xy=(0.5, 0.5),
                    xycoords="axes fraction", ha="center", fontsize=14,
                    fontweight="bold", color=TEAL)

    # NCC bar
    ax_ncc = fig.add_subplot(gs[0, 1])
    bars = ax_ncc.bar(["Before", "After"], [ncc_before, ncc_after],
                      color=[CORAL, TEAL], width=0.5, alpha=0.85)
    ax_ncc.set_ylabel("NCC")
    ax_ncc.set_title("Normalised Cross-Correlation", fontweight="bold")
    ax_ncc.set_ylim(min(ncc_before, ncc_after) * 0.998, 1.001)
    for bar, val in zip(bars, [ncc_before, ncc_after]):
        ax_ncc.text(bar.get_x() + bar.get_width() / 2.,
                    bar.get_height() * 1.0001,
                    f"{val:.6f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")

    # SSIM bar
    ax_ssim_bar = fig.add_subplot(gs[0, 2])
    bars = ax_ssim_bar.bar(["Before", "After"],
                           [mean_ssim_before, mean_ssim_after],
                           color=[CORAL, TEAL], width=0.5, alpha=0.85)
    ax_ssim_bar.set_ylabel("SSIM")
    ax_ssim_bar.set_title("Structural Similarity (mean)", fontweight="bold")
    ax_ssim_bar.set_ylim(min(mean_ssim_before, mean_ssim_after) * 0.95,
                         max(mean_ssim_before, mean_ssim_after) * 1.05)
    for bar, val in zip(bars, [mean_ssim_before, mean_ssim_after]):
        ax_ssim_bar.text(bar.get_x() + bar.get_width() / 2.,
                         bar.get_height() * 1.002,
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10,
                         fontweight="bold")

    # Slice-by-slice SSIM
    z_range = np.arange(fixed.shape[0])

    ax_ssim_sl = fig.add_subplot(gs[1, :2])
    ax_ssim_sl.fill_between(z_range, ssim_before, alpha=0.15, color=CORAL)
    ax_ssim_sl.plot(z_range, ssim_before, color=CORAL, linewidth=1.5,
                    label=f"Before (mean={mean_ssim_before:.4f})", alpha=0.8)
    ax_ssim_sl.fill_between(z_range, ssim_after, alpha=0.15, color=TEAL)
    ax_ssim_sl.plot(z_range, ssim_after, color=TEAL, linewidth=1.5,
                    label=f"After (mean={mean_ssim_after:.4f})", alpha=0.8)
    ax_ssim_sl.set_xlabel("Z slice", fontsize=11)
    ax_ssim_sl.set_ylabel("SSIM", fontsize=11)
    ax_ssim_sl.set_title("SSIM per Axial Slice", fontweight="bold")
    ax_ssim_sl.legend(fontsize=10, loc="lower right")
    ax_ssim_sl.set_xlim(0, len(z_range) - 1)
    ax_ssim_sl.grid(True, alpha=0.3)

    # Slice-by-slice NCC
    ax_ncc_sl = fig.add_subplot(gs[1, 2])
    ax_ncc_sl.fill_between(z_range, ncc_before_sl, alpha=0.15, color=CORAL)
    ax_ncc_sl.plot(z_range, ncc_before_sl, color=CORAL, linewidth=1.5,
                   label="Before", alpha=0.8)
    ax_ncc_sl.fill_between(z_range, ncc_after_sl, alpha=0.15, color=TEAL)
    ax_ncc_sl.plot(z_range, ncc_after_sl, color=TEAL, linewidth=1.5,
                   label="After", alpha=0.8)
    ax_ncc_sl.set_xlabel("Z slice", fontsize=11)
    ax_ncc_sl.set_ylabel("NCC", fontsize=11)
    ax_ncc_sl.set_title("NCC per Axial Slice", fontweight="bold")
    ax_ncc_sl.legend(fontsize=9, loc="lower right")
    ax_ncc_sl.set_xlim(0, len(z_range) - 1)
    ax_ncc_sl.grid(True, alpha=0.3)

    fig.savefig(os.path.join(RESULTS_DIR, "fig5_metrics.png"))
    plt.close(fig)
    print("  Saved fig5_metrics.png")

    return {
        "MSE_before": mse_before,  "MSE_after": mse_after,
        "NCC_before": ncc_before,  "NCC_after": ncc_after,
        "SSIM_before": mean_ssim_before, "SSIM_after": mean_ssim_after,
    }


# ─── Figure 6 ────────────────────────────────────────────────────────────────

def fig6_histogram(fixed, moving, registered, mask=None):
    """Difference histograms with robust statistics (P99.9).

    Reports percentile-based max instead of raw max to avoid the
    single-voxel resampling outlier that inflates |max| after
    registration.
    """
    print("Generating Fig 6: Difference histogram ...")

    diff_before = (fixed - moving).ravel()
    diff_after  = (fixed - registered).ravel()

    # If mask provided, also compute inside-specimen stats
    if mask is not None:
        diff_before_m = (fixed[mask] - moving[mask])
        diff_after_m  = (fixed[mask] - registered[mask])
    else:
        diff_before_m = diff_before
        diff_after_m  = diff_after

    bins = np.linspace(-0.4, 0.4, 301)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Distribution of Voxel Intensity Differences",
                 fontsize=16, fontweight="bold", color=SLATE, y=1.02)

    for ax, diff_all, diff_m, title, colour in [
        (ax1, diff_before, diff_before_m, "Before Registration", CORAL),
        (ax2, diff_after,  diff_after_m,  "After Registration",  TEAL),
    ]:
        ax.hist(diff_all, bins=bins, density=True, color=colour, alpha=0.7,
                edgecolor="none")
        ax.set_title(title, fontweight="bold", color=colour)
        ax.set_xlabel("Fixed − Compared")
        ax.set_ylabel("Density")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlim(-0.4, 0.4)

        p999 = np.percentile(np.abs(diff_m), 99.9)
        stats = (f"μ = {diff_m.mean():.5f}\n"
                 f"σ = {diff_m.std():.5f}\n"
                 f"P99.9 |diff| = {p999:.4f}\n"
                 f"|max| = {np.abs(diff_all).max():.4f}")
        if mask is not None:
            stats += "\n(stats inside specimen)"
        ax.text(0.97, 0.95, stats, transform=ax.transAxes, fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.9))

    ymax = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(0, ymax)
    ax2.set_ylim(0, ymax)

    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "fig6_histogram.png"),
               bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig6_histogram.png")


# ─── Figure 7 ────────────────────────────────────────────────────────────────

def fig7_edge_detail(fixed, moving, registered):
    """Zoomed ROIs at specimen edges/notches."""
    print("Generating Fig 7: Edge detail comparison ...")
    nz, ny, nx = fixed.shape
    z_mid = nz // 2

    rois = [
        ("Top edge",      z_mid,    (10,  90,  50, 220)),
        ("Right edge",    z_mid,    (60, 210, 180, 270)),
        ("Bottom edge",   z_mid,    (190, 265, 40, 230)),
        ("Notch region",  nz // 4,  (50, 200,  10, 130)),
    ]
    vmax = 0.85

    fig, axes = plt.subplots(len(rois), 4, figsize=(20, 5 * len(rois)))
    fig.suptitle("Edge & Notch Detail - Registration Quality",
                 fontsize=18, fontweight="bold", color=SLATE, y=0.99)

    for row, (roi_name, z, (y1, y2, x1, x2)) in enumerate(rois):
        f_sl = fixed[z, y1:y2, x1:x2]
        m_sl = moving[z, y1:y2, x1:x2]
        r_sl = registered[z, y1:y2, x1:x2]

        axes[row, 0].imshow(f_sl, cmap="gray", vmin=0, vmax=vmax)
        if row == 0:
            axes[row, 0].set_title("Fixed", fontsize=12, color=SLATE)
        axes[row, 0].set_ylabel(f"{roi_name}\n(Z={z})", fontsize=11,
                                fontweight="bold", color=TEAL)

        f_n = np.clip(f_sl / vmax, 0, 1)
        m_n = np.clip(m_sl / vmax, 0, 1)
        r_n = np.clip(r_sl / vmax, 0, 1)

        axes[row, 1].imshow(np.clip(np.stack([f_n, m_n, f_n], axis=-1), 0, 1))
        if row == 0:
            axes[row, 1].set_title("Before Reg. (overlay)", fontsize=12,
                                   color=CORAL)

        axes[row, 2].imshow(np.clip(np.stack([f_n, r_n, f_n], axis=-1), 0, 1))
        if row == 0:
            axes[row, 2].set_title("After Reg. (overlay)", fontsize=12,
                                   color=TEAL)

        diff = np.abs(f_sl - r_sl)
        im = axes[row, 3].imshow(diff, cmap="hot", vmin=0,
                                 vmax=max(0.1, diff.max()))
        if row == 0:
            axes[row, 3].set_title("|Fixed − Registered|", fontsize=12,
                                   color=SLATE)
        divider = make_axes_locatable(axes[row, 3])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        for ax in axes[row, :]:
            ax.set_xticks([]); ax.set_yticks([])

    plt.subplots_adjust(wspace=0.08, hspace=0.15)
    fig.savefig(os.path.join(RESULTS_DIR, "fig7_edge_detail.png"))
    plt.close(fig)
    print("  Saved fig7_edge_detail.png")


# ─── Figure 8 ────────────────────────────────────────────────────────────────

def fig8_summary(fixed, moving, registered, metrics):
    """Single-page dashboard combining key visuals and metrics table."""
    print("Generating Fig 8: Summary dashboard ...")
    nz, ny, nx = fixed.shape
    z_mid = nz // 2
    vmax = 0.85

    f_sl = fixed[z_mid]
    m_sl = moving[z_mid]
    r_sl = registered[z_mid]

    f_n = np.clip(f_sl / vmax, 0, 1)
    m_n = np.clip(m_sl / vmax, 0, 1)
    r_n = np.clip(r_sl / vmax, 0, 1)

    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.3,
                           height_ratios=[1.2, 1, 0.8])

    # Row 0
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(f_n, cmap="gray"); ax0.axis("off")
    ax0.set_title("Fixed\n(Before deformation)", fontweight="bold", fontsize=11)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(np.clip(np.stack([f_n, m_n, f_n], axis=-1), 0, 1)); ax1.axis("off")
    ax1.set_title("Misalignment\n(before registration)", fontweight="bold",
                  fontsize=11, color=CORAL)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(np.clip(np.stack([f_n, r_n, f_n], axis=-1), 0, 1)); ax2.axis("off")
    ax2.set_title("After Registration\n(aligned)", fontweight="bold",
                  fontsize=11, color=TEAL)

    ax3 = fig.add_subplot(gs[0, 3])
    diff = np.abs(f_sl - r_sl)
    ax3.imshow(diff, cmap="hot", vmin=0, vmax=max(0.1, diff.max())); ax3.axis("off")
    ax3.set_title("Residual difference\n|Fixed − Registered|", fontweight="bold",
                  fontsize=11)

    # Row 1: coronal + sagittal overlays
    for col, (view, idx, slicer) in enumerate([
        ("Coronal",  ny // 2, lambda v, i: v[:, i, :]),
        ("Sagittal", nx // 2, lambda v, i: v[:, :, i]),
    ]):
        f_v = np.clip(slicer(fixed, idx) / vmax, 0, 1)
        m_v = np.clip(slicer(moving, idx) / vmax, 0, 1)
        r_v = np.clip(slicer(registered, idx) / vmax, 0, 1)

        ax_b = fig.add_subplot(gs[1, col * 2])
        ax_b.imshow(np.clip(np.stack([f_v, m_v, f_v], axis=-1), 0, 1))
        ax_b.set_title(f"{view} - Before", fontsize=10, color=CORAL)
        ax_b.axis("off")

        ax_a = fig.add_subplot(gs[1, col * 2 + 1])
        ax_a.imshow(np.clip(np.stack([f_v, r_v, f_v], axis=-1), 0, 1))
        ax_a.set_title(f"{view} - After", fontsize=10, color=TEAL)
        ax_a.axis("off")

    # Row 2: metrics table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")

    table_data = [
        ["MSE",  f'{metrics["MSE_before"]:.6f}',  f'{metrics["MSE_after"]:.6f}',
         f'↓ {(1 - metrics["MSE_after"] / metrics["MSE_before"]) * 100:.1f}%'],
        ["NCC",  f'{metrics["NCC_before"]:.6f}',  f'{metrics["NCC_after"]:.6f}',
         f'↑ {(metrics["NCC_after"] / metrics["NCC_before"] - 1) * 100:.2f}%'],
        ["SSIM", f'{metrics["SSIM_before"]:.4f}', f'{metrics["SSIM_after"]:.4f}',
         f'↑ {(metrics["SSIM_after"] / metrics["SSIM_before"] - 1) * 100:.1f}%'],
    ]

    table = ax_table.table(
        cellText=table_data,
        colLabels=["Metric", "Before Registration", "After Registration",
                   "Improvement"],
        loc="center", cellLoc="center",
        colColours=[SLATE, CORAL, TEAL, "#95a5a6"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(0.8, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor("#cccccc")

    fig.suptitle("3D CT Registration - Summary Dashboard",
                 fontsize=20, fontweight="bold", color=SLATE, y=1.0)
    fig.savefig(os.path.join(RESULTS_DIR, "fig8_summary_dashboard.png"),
               bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig8_summary_dashboard.png")


# ─── Figure 9: Masked deformation analysis ──────────────────────────────────

def fig9_masked_deformation(fixed, registered, mask, mask_eroded, k_sigma=3.0):
    """Residual difference inside specimen + thresholded change map.

    Column 0 : |diff| inside mask (background transparent)
    Column 1 : thresholded change map (|diff| > k*sigma) on full mask
    Column 2 : same on eroded mask (avoids boundary artefacts)

    Returns
    -------
    dict - quantitative change-map statistics (for the report)
    """
    print("Generating Fig 9: Masked deformation analysis ...")
    nz, ny, nx = fixed.shape
    z_mid = nz // 2

    diff_vol = np.abs(fixed - registered)

    # Statistics inside eroded mask for thresholding
    interior_vals = diff_vol[mask_eroded]
    sigma = float(interior_vals.std())
    thresh = k_sigma * sigma
    print(f"  Change threshold: {k_sigma}σ = {thresh:.5f}")

    # ── Quantitative change-map statistics (eroded mask) ─────────────
    change_vol = (diff_vol > thresh) & mask_eroded
    n_changed   = int(change_vol.sum())
    n_eroded    = int(mask_eroded.sum())
    pct_changed = 100.0 * n_changed / n_eroded if n_eroded else 0.0

    labelled, n_components = ndimage.label(change_vol)
    if n_components > 0:
        comp_sizes = ndimage.sum(change_vol, labelled, range(1, n_components + 1))
        largest_comp = int(comp_sizes.max())
    else:
        largest_comp = 0

    # Same stats on full (non-eroded) mask to show boundary effect
    change_vol_full = (diff_vol > thresh) & mask
    n_full_mask     = int(mask.sum())
    n_changed_full  = int(change_vol_full.sum())
    pct_changed_full = 100.0 * n_changed_full / n_full_mask if n_full_mask else 0.0

    change_stats = {
        "threshold_k_sigma":   k_sigma,
        "threshold_value":     thresh,
        "full_mask_voxels":    n_full_mask,
        "changed_voxels_full": n_changed_full,
        "changed_pct_full":    pct_changed_full,
        "eroded_mask_voxels":  n_eroded,
        "changed_voxels":      n_changed,
        "changed_pct":         pct_changed,
        "n_components":        n_components,
        "largest_component":   largest_comp,
    }
    print(f"  Changed voxels (full mask):   {n_changed_full:,} / {n_full_mask:,} "
          f"({pct_changed_full:.2f}%)")
    print(f"  Changed voxels (eroded mask): {n_changed:,} / {n_eroded:,} "
          f"({pct_changed:.2f}%)")
    print(f"  Connected components: {n_components}, "
          f"largest = {largest_comp:,} voxels")

    # ── Figure ───────────────────────────────────────────────────────
    views = [
        ("Axial",    z_mid,    lambda v, i: v[i, :, :]),
        ("Coronal",  ny // 2, lambda v, i: v[:, i, :]),
        ("Sagittal", nx // 2, lambda v, i: v[:, :, i]),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f"Deformation Analysis Inside Specimen (threshold = {k_sigma}σ)",
                 fontsize=18, fontweight="bold", color=SLATE, y=0.98)

    for row, (view_name, idx, slicer) in enumerate(views):
        diff_sl = slicer(diff_vol, idx)
        mask_sl = slicer(mask.astype(float), idx) > 0.5
        mask_ero_sl = slicer(mask_eroded.astype(float), idx) > 0.5

        # Col 0: masked |diff|
        masked_diff = np.where(mask_sl, diff_sl, np.nan)
        im0 = axes[row, 0].imshow(masked_diff, cmap="inferno", vmin=0,
                                  vmax=max(0.05, np.nanmax(masked_diff)))
        if row == 0:
            axes[row, 0].set_title("|Fixed − Registered|\n(specimen mask)",
                                   fontsize=12, color=TEAL)
        axes[row, 0].set_ylabel(f"{view_name} (slice {idx})",
                                fontsize=11, fontweight="bold", color=TEAL)
        divider0 = make_axes_locatable(axes[row, 0])
        plt.colorbar(im0, cax=divider0.append_axes("right", size="4%", pad=0.05))

        # Col 1: thresholded change map on full mask
        change = (diff_sl > thresh) & mask_sl
        axes[row, 1].imshow(slicer(fixed, idx), cmap="gray", vmin=0, vmax=0.85)
        change_rgba = np.zeros((*diff_sl.shape, 4))
        change_rgba[change, 0] = 1.0   # red
        change_rgba[change, 3] = 0.6   # alpha
        axes[row, 1].imshow(change_rgba)
        if row == 0:
            axes[row, 1].set_title(f"Change map (|diff| > {k_sigma}σ)\n"
                                   f"on specimen mask",
                                   fontsize=12, color=CORAL)

        # Col 2: thresholded change map on eroded mask
        change_ero = (diff_sl > thresh) & mask_ero_sl
        axes[row, 2].imshow(slicer(fixed, idx), cmap="gray", vmin=0, vmax=0.85)
        change_rgba2 = np.zeros((*diff_sl.shape, 4))
        change_rgba2[change_ero, 0] = 1.0
        change_rgba2[change_ero, 3] = 0.6
        axes[row, 2].imshow(change_rgba2)
        if row == 0:
            axes[row, 2].set_title(f"Change map (|diff| > {k_sigma}σ)\n"
                                   f"on eroded mask",
                                   fontsize=12, color=SLATE)

        for ax in axes[row, :]:
            ax.set_xticks([]); ax.set_yticks([])

    # Interpretive footnote
    fig.text(
        0.5, 0.01,
        f"Note: {pct_changed:.1f}% of eroded-mask voxels exceed the {k_sigma}σ threshold "
        f"({pct_changed_full:.1f}% on the full mask - the gap reflects boundary artefacts).\n"
        "This percentage is an upper bound on potential deformation-related change because "
        "acquisition artefacts (e.g., stripe/ring effects) may contribute.",
        ha="center", fontsize=10, style="italic", color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f9f9f9", ec="#cccccc"),
    )

    plt.subplots_adjust(wspace=0.15, hspace=0.15, bottom=0.07)
    fig.savefig(os.path.join(RESULTS_DIR, "fig9_masked_deformation.png"))
    plt.close(fig)
    print("  Saved fig9_masked_deformation.png")

    return change_stats


# ─── Figure 10: Registration settings ────────────────────────────────────────

def fig10_settings():
    """Single-page table listing all key registration parameters."""
    print("Generating Fig 10: Registration settings ...")

    settings = [
        ["Transform model",     "Euler3DTransform (rigid: 3 rotations + 3 translations)"],
        ["Initialisation",      "Centre-of-mass (MOMENTS)"],
        ["Similarity metric",   "Mattes Mutual Information (50 histogram bins)"],
        ["Metric sampling",     "RANDOM, 15 % of voxels per evaluation"],
        ["Optimiser",           "Regular Step Gradient Descent"],
        ["Learning rate",       "1.0 (initial)"],
        ["Min step size",       "1 × 10⁻⁴"],
        ["Relaxation factor",   "0.5"],
        ["Max iterations/level","500"],
        ["Gradient tolerance",  "1 × 10⁻⁸"],
        ["Scale estimation",    "Physical shift (automatic)"],
        ["Multi-resolution",    "3 levels"],
        ["Shrink factors",      "4 → 2 → 1"],
        ["Smoothing sigmas",    "2.0 → 1.0 → 0.0  (physical units)"],
        ["Interpolator",        "Linear (registration) / Linear (resampling)"],
        ["Preprocessing",       "Cast to float32, isotropic 1×1×1 spacing"],
        ["Mask (metrics)",      "Otsu threshold + morphological cleanup; eroded 5 px"],
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")
    fig.suptitle("Registration Settings",
                 fontsize=20, fontweight="bold", color=SLATE, y=0.96)

    table = ax.table(
        cellText=settings,
        colLabels=["Parameter", "Value"],
        loc="center", cellLoc="left",
        colColours=[TEAL, SLATE],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.auto_set_column_width([0, 1])
    table.scale(1.0, 1.8)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_edgecolor("white")
        else:
            cell.set_edgecolor("#dddddd")
            if col == 0:
                cell.set_text_props(fontweight="bold", color=SLATE)

    fig.savefig(os.path.join(RESULTS_DIR, "fig10_registration_settings.png"))
    plt.close(fig)
    print("  Saved fig10_registration_settings.png")

#!/usr/bin/env python3
"""
Entry point: generate all 10 presentation-quality figures.

Requires ``run_registration.py`` to have been executed first
(needs the registered volume in ``results/``).
"""

import os

from ct_registration.config import RESULTS_DIR
from ct_registration.io import load_all_numpy
from ct_registration.masking import create_specimen_mask
from ct_registration.figures import (
    fig1_overview, fig2_misalignment, fig3_difference_maps,
    fig4_checkerboard, fig5_metrics, fig6_histogram,
    fig7_edge_detail, fig8_summary,
    fig9_masked_deformation, fig10_settings,
)
from ct_registration.report import append_change_stats


# Old quick-preview files superseded by the presentation figures
_OLD_FIGURES = [
    "checkerboard.png", "difference_histogram.png",
    "misalignment_after.png", "misalignment_before.png",
    "slices_axial.png", "slices_coronal.png", "slices_sagittal.png",
]


def main():
    print("=" * 60)
    print("  Generating Presentation-Quality Visualizations")
    print("=" * 60 + "\n")

    fixed, moving, registered = load_all_numpy()

    # Build specimen mask from fixed volume
    print("\n── Building Specimen Mask ──")
    mask, mask_eroded = create_specimen_mask(fixed)

    # Remove old quick-preview figures if present
    for f in _OLD_FIGURES:
        p = os.path.join(RESULTS_DIR, f)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed old: {f}")

    print()
    fig1_overview(fixed, moving, registered)
    fig2_misalignment(fixed, moving, registered)
    fig3_difference_maps(fixed, moving, registered, mask=mask)
    fig4_checkerboard(fixed, registered)
    metrics = fig5_metrics(fixed, moving, registered)
    fig6_histogram(fixed, moving, registered, mask=mask)
    fig7_edge_detail(fixed, moving, registered)
    fig8_summary(fixed, moving, registered, metrics)
    change_stats = fig9_masked_deformation(fixed, registered, mask, mask_eroded)
    fig10_settings()

    # Append change-map statistics to the text report
    append_change_stats(change_stats)

    print(f"\n{'=' * 60}")
    print(f"  All 10 figures saved to: {RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

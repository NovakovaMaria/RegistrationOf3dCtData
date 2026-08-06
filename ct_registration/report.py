"""
Save a text report of the registration metrics and transform parameters.
"""

import os
import SimpleITK as sitk

from .config import RESULTS_DIR


def save_metrics_report(metrics: dict, transform: sitk.Transform) -> str:
    """
    Write ``registration_report.txt`` to *RESULTS_DIR*.

    Parameters
    ----------
    metrics : dict  – output of ``metrics.quantitative_comparison()``
    transform : sitk.Transform

    Returns
    -------
    str – path to the written report
    """
    path = os.path.join(RESULTS_DIR, "registration_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Registration of 3D CT Data - Results Report\n")
        f.write("=" * 60 + "\n\n")

        # Settings section
        f.write("─" * 60 + "\n")
        f.write("  Registration Settings\n")
        f.write("─" * 60 + "\n")
        f.write("Registration method:  Rigid (Euler3DTransform)\n")
        f.write("Initialisation:       Centre-of-mass (MOMENTS)\n")
        f.write("Similarity metric:    Mattes Mutual Information (50 bins)\n")
        f.write("Metric sampling:      RANDOM, 15%\n")
        f.write("Optimiser:            Regular Step Gradient Descent\n")
        f.write("  Learning rate:      1.0\n")
        f.write("  Min step:           1e-4\n")
        f.write("  Relaxation factor:  0.5\n")
        f.write("  Max iterations:     500 per level\n")
        f.write("  Gradient tolerance: 1e-8\n")
        f.write("Multi-resolution:     3 levels (shrink 4→2→1)\n")
        f.write("Smoothing sigmas:     2.0 → 1.0 → 0.0 (physical units)\n")
        f.write("Interpolator:         Linear\n")
        f.write("Preprocessing:        float32 cast, 1×1×1 spacing\n")
        f.write("Specimen mask:        Otsu + morph. cleanup; eroded 5 px\n\n")

        f.write(f"Final transform:\n  {transform}\n\n")

        # Metrics section
        f.write("─" * 60 + "\n")
        f.write("  Quantitative Metrics\n")
        f.write("─" * 60 + "\n")
        for label, vals in metrics.items():
            f.write(f"\n{label}:\n")
            for k, v in vals.items():
                f.write(f"  {k:25s} = {v:.6f}\n")

    print(f"\n  Saved {path}")
    return path


def append_change_stats(change_stats: dict) -> None:
    """Append change-map statistics to an existing registration report."""
    path = os.path.join(RESULTS_DIR, "registration_report.txt")
    with open(path, "a") as f:
        f.write("\n" + "─" * 60 + "\n")
        f.write("  Change-Map Analysis\n")
        f.write("─" * 60 + "\n")
        f.write(f"  Threshold:             {change_stats['threshold_k_sigma']}σ "
                f"= {change_stats['threshold_value']:.6f}\n")
        f.write(f"\n  Full (non-eroded) mask:\n")
        f.write(f"    Mask voxels:         {change_stats['full_mask_voxels']:,}\n")
        f.write(f"    Voxels above thresh: {change_stats['changed_voxels_full']:,}\n")
        f.write(f"    % above threshold:   {change_stats['changed_pct_full']:.2f}%\n")
        f.write(f"\n  Eroded mask (5 px inward):\n")
        f.write(f"    Mask voxels:         {change_stats['eroded_mask_voxels']:,}\n")
        f.write(f"    Voxels above thresh: {change_stats['changed_voxels']:,}\n")
        f.write(f"    % above threshold:   {change_stats['changed_pct']:.2f}%\n")
        f.write(f"\n  Connected components:  {change_stats['n_components']}\n")
        f.write(f"  Largest component:     {change_stats['largest_component']:,} voxels\n")
        f.write(f"\n  The eroded-mask percentage ({change_stats['changed_pct']:.2f}%) is an "
                "upper bound on\n"
                "  potential deformation-related change because acquisition\n"
                "  artefacts (e.g., stripe/ring effects) may contribute. The\n"
                f"  higher full-mask value ({change_stats['changed_pct_full']:.2f}%) "
                "confirms that specimen\n"
                "  boundaries inflate the raw count.\n")
    print(f"  Appended change-map stats to {path}")

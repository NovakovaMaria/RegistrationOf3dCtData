"""
Save a text report of the registration metrics and transform parameters.
"""

import os
import SimpleITK as sitk

from .config import RESULTS_DIR


def save_metrics_report(metrics: dict, transform: sitk.Transform) -> str:
    """
    Write registration_report.txt to RESULTS_DIR.

    Parameters
    ----------
    metrics   : dict            – {label: {metric_name: value}}
    transform : sitk.Transform  – final registration transform

    Returns
    -------
    str – path to the written report
    """
    path = os.path.join(RESULTS_DIR, "registration_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Registration of 3D CT Data — Results Report\n")
        f.write("=" * 60 + "\n\n")

        # Settings
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
        f.write("Multi-resolution:     3 levels (shrink 4→2→1)\n")
        f.write("Smoothing sigmas:     2.0 → 1.0 → 0.0 (physical units)\n")
        f.write("Interpolator:         Linear\n\n")

        f.write(f"Final transform:\n  {transform}\n\n")

        # Metrics
        f.write("─" * 60 + "\n")
        f.write("  Quantitative Metrics\n")
        f.write("─" * 60 + "\n")
        for label, vals in metrics.items():
            f.write(f"\n{label}:\n")
            for k, v in vals.items():
                f.write(f"  {k:25s} = {v:.6f}\n")

    print(f"\n  Saved report → {path}")
    return path
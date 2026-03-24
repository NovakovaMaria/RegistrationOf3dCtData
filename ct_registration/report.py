"""
Save a text report of the registration metrics.
"""

import os
from .config import RESULTS_DIR


def save_metrics_report(metrics: dict) -> str:
    """
    Write registration_report.txt to RESULTS_DIR.

    Parameters
    ----------
    metrics : dict  – {label: {metric_name: value}}

    Returns
    -------
    str – path to the written report
    """
    path = os.path.join(RESULTS_DIR, "registration_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  Registration of 3D CT Data — Results Report\n")
        f.write("=" * 60 + "\n\n")
        for label, vals in metrics.items():
            f.write(f"{label}:\n")
            for k, v in vals.items():
                f.write(f"  {k:25s} = {v:.6f}\n")
            f.write("\n")

    print(f"\n  Saved report into {path}")
    return path

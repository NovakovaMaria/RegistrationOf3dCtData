#!/usr/bin/env python3
"""
Entry point: run the rigid-registration pipeline.
"""

import numpy as np

from ct_registration.io import load_volumes_sitk, sitk_to_numpy, save_registered_volume
from ct_registration.registration import rigid_register, resample
from ct_registration.metrics import quantitative_comparison
from ct_registration.masking import create_specimen_mask
from ct_registration.visualization import (
    plot_central_slices, plot_checkerboard,
    plot_misalignment_overlay, plot_difference_histogram,
)
from ct_registration.report import save_metrics_report
from ct_registration.config import RESULTS_DIR


def main():
    print("=" * 60)
    print("  Registration of 3D CT Data")
    print("=" * 60)

    # Load
    fixed_sitk, moving_sitk = load_volumes_sitk()

    # Register
    transform = rigid_register(fixed_sitk, moving_sitk)

    # Resample and save
    registered_sitk = resample(fixed_sitk, moving_sitk, transform)
    fixed_arr = sitk_to_numpy(fixed_sitk)
    moving_arr = sitk_to_numpy(moving_sitk)
    registered_arr = sitk_to_numpy(registered_sitk)
    save_registered_volume(registered_arr)

    # Build specimen mask
    print("\n── Building Specimen Mask ──")
    fmax = max(fixed_arr.max(), moving_arr.max())
    mask, mask_eroded = create_specimen_mask(
        fixed_arr.astype(np.float64) / fmax
    )

    metrics = quantitative_comparison(
        fixed_arr, moving_arr, registered_arr,
        mask=mask, mask_eroded=mask_eroded,
    )

    # Save report
    save_metrics_report(metrics, transform)

    # Visualisation
    plot_central_slices(fixed_arr, moving_arr, registered_arr)
    plot_checkerboard(fixed_arr, registered_arr)
    plot_misalignment_overlay(fixed_arr, moving_arr, registered_arr)
    plot_difference_histogram(fixed_arr, moving_arr, registered_arr)

    print("\nDone.")


if __name__ == "__main__":
    main()
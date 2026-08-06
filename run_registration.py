#!/usr/bin/env python3
"""
Entry point: run the full rigid-registration pipeline.

Steps:
  1. Load TIF stacks (fixed + moving)
  2. Rigid registration (Euler3D, Mattes MI, 3-level multi-resolution)
  3. Resample moving image onto fixed grid
  4. Save registered volume as TIF
  5. Build specimen mask (Otsu + morphology)
  6. Quantitative comparison (whole-volume + masked + eroded)
  7. Quick-preview visualisations
  8. Text report
"""

import numpy as np

from ct_registration.io import (
    load_volumes_sitk, sitk_to_numpy, save_registered_volume,
)
from ct_registration.registration import rigid_register, resample
from ct_registration.masking import create_specimen_mask
from ct_registration.metrics import quantitative_comparison
from ct_registration.visualization import (
    plot_central_slices, plot_checkerboard,
    plot_misalignment_overlay, plot_difference_histogram,
)
from ct_registration.report import save_metrics_report


def main():
    print("=" * 60)
    print("  Registration of 3D CT Data")
    print("=" * 60)

    # 1. Load
    fixed_sitk, moving_sitk = load_volumes_sitk()

    # 2. Register
    transform = rigid_register(fixed_sitk, moving_sitk)

    # 3. Resample
    registered_sitk = resample(fixed_sitk, moving_sitk, transform)

    # 4. Convert to NumPy & save registered volume
    fixed_arr = sitk_to_numpy(fixed_sitk)
    moving_arr = sitk_to_numpy(moving_sitk)
    registered_arr = sitk_to_numpy(registered_sitk)
    save_registered_volume(registered_arr)

    # 5. Build specimen mask
    print("\n── Building Specimen Mask ──")
    gmax = max(fixed_arr.max(), moving_arr.max())
    mask, mask_eroded = create_specimen_mask(
        fixed_arr.astype(np.float64) / gmax
    )

    # 6. Quantitative comparison (whole-volume + masked + eroded)
    metrics = quantitative_comparison(
        fixed_arr, moving_arr, registered_arr,
        mask=mask, mask_eroded=mask_eroded,
    )

    # 7. Quick-preview visualisations
    plot_central_slices(fixed_arr, moving_arr, registered_arr)
    plot_checkerboard(fixed_arr, registered_arr)
    plot_misalignment_overlay(fixed_arr, moving_arr, registered_arr)
    plot_difference_histogram(fixed_arr, moving_arr, registered_arr)

    # 8. Text report
    save_metrics_report(metrics, transform)

    print("\n" + "=" * 60)
    print("  All done!  Results saved to: results/")
    print("=" * 60)


if __name__ == "__main__":
    main()

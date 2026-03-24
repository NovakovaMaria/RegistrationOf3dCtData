#!/usr/bin/env python3
"""
Entry point: run the rigid-registration pipeline.
"""

import numpy as np

from ct_registration.io import load_volumes_sitk, sitk_to_numpy, save_registered_volume
from ct_registration.registration import rigid_register, resample
from ct_registration.metrics import compute_mse, compute_ncc, compute_masked_metrics, compute_ssim_per_slice
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

    # Metrics
    fmax = max(fixed_arr.max(), moving_arr.max())
    f = fixed_arr.astype(np.float64) / fmax
    m = moving_arr.astype(np.float64) / fmax
    r = registered_arr.astype(np.float64) / fmax

    # Build specimen mask
    print("\n── Building Specimen Mask ──")
    mask, mask_eroded = create_specimen_mask(f)

    print("\n── Quantitative Comparison ──")
    print(f"  Whole volume — Before: MSE={compute_mse(f, m):.6f}  NCC={compute_ncc(f, m):.6f}  SSIM={compute_ssim_per_slice(f, m):.6f}")
    print(f"  Whole volume — After:  MSE={compute_mse(f, r):.6f}  NCC={compute_ncc(f, r):.6f}  SSIM={compute_ssim_per_slice(f, r):.6f}")
    before_m = compute_masked_metrics(f, m, mask)
    after_m  = compute_masked_metrics(f, r, mask)
    print(f"  Masked       — Before: MSE={before_m['MSE']:.6f}  NCC={before_m['NCC']:.6f}")
    print(f"  Masked       — After:  MSE={after_m['MSE']:.6f}  NCC={after_m['NCC']:.6f}")
    before_e = compute_masked_metrics(f, m, mask_eroded)
    after_e  = compute_masked_metrics(f, r, mask_eroded)
    print(f"  Eroded mask  — Before: MSE={before_e['MSE']:.6f}  NCC={before_e['NCC']:.6f}")
    print(f"  Eroded mask  — After:  MSE={after_e['MSE']:.6f}  NCC={after_e['NCC']:.6f}")

    # Save report
    save_metrics_report({
        "Before registration (whole volume)": {"MSE": compute_mse(f, m), "NCC": compute_ncc(f, m), "SSIM": compute_ssim_per_slice(f, m)},
        "After registration (whole volume)":  {"MSE": compute_mse(f, r), "NCC": compute_ncc(f, r), "SSIM": compute_ssim_per_slice(f, r)},
        "Before registration (masked)":       before_m,
        "After registration (masked)":        after_m,
    })

    # Visualisation
    plot_central_slices(f, m, r)
    plot_checkerboard(f, r)
    plot_misalignment_overlay(f, m, r)
    plot_difference_histogram(f, m, r)

    print("\nDone.")


if __name__ == "__main__":
    main()
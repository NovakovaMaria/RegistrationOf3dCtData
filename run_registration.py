#!/usr/bin/env python3
"""
Entry point: run the rigid-registration pipeline.
"""

from ct_registration.io import load_volumes_sitk, sitk_to_numpy, save_registered_volume
from ct_registration.registration import rigid_register, resample
from ct_registration.metrics import compute_mse, compute_ncc


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
      print("\n── Quantitative Comparison ──")
      print(f"  Before — MSE: {compute_mse(fixed_arr, moving_arr):.6f}  "
            f"NCC: {compute_ncc(fixed_arr, moving_arr):.6f}")
      print(f"  After  — MSE: {compute_mse(fixed_arr, registered_arr):.6f}  "
            f"NCC: {compute_ncc(fixed_arr, registered_arr):.6f}")

      print("\nDone.")


if __name__ == "__main__":
      main()

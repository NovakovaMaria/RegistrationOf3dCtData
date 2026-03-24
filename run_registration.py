#!/usr/bin/env python3
"""
Entry point: run the rigid-registration pipeline.
"""

from ct_registration.io import load_volumes_sitk, sitk_to_numpy, save_registered_volume
from ct_registration.registration import rigid_register, resample


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
    registered_arr = sitk_to_numpy(registered_sitk)
    save_registered_volume(registered_arr)

    print("\nDone.")


if __name__ == "__main__":
    main()

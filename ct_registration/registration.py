"""
Rigid registration of 3-D CT volumes using SimpleITK.
"""

import SimpleITK as sitk


def _registration_callback(method):
    """Print optimiser progress (called each iteration)."""
    print(f"    Iter {method.GetOptimizerIteration():4d}  "
          f"Metric = {method.GetMetricValue():.6f}")


def rigid_register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """
    Rigid (Euler3D) registration with Mattes Mutual Information.

    Uses a 3-level multi-resolution pyramid (shrink 4→2→1) and
    Regular Step Gradient Descent optimisation.

    Parameters
    ----------
    fixed : sitk.Image   – reference volume (before deformation)
    moving : sitk.Image  – volume to align  (after deformation)

    Returns
    -------
    sitk.Transform – the optimised rigid transform
    """
    print("\n── Rigid Registration ──")

    # Centre-of-mass initialisation
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    print(f"  Initial transform (centre-of-mass alignment):\n"
          f"    {initial_transform}")

    reg = sitk.ImageRegistrationMethod()

    # Similarity metric
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)

    # Interpolator
    reg.SetInterpolator(sitk.sitkLinear)

    # Optimiser
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=500,
        gradientMagnitudeTolerance=1e-8,
        relaxationFactor=0.5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution: 3 levels
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2.0, 1.0, 0.0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Initial transform
    reg.SetInitialTransform(initial_transform, inPlace=False)

    # Callbacks
    reg.AddCommand(sitk.sitkIterationEvent,
                    lambda: _registration_callback(reg))
    reg.AddCommand(sitk.sitkStartEvent,
                    lambda: print("  Registration started ..."))
    reg.AddCommand(sitk.sitkEndEvent,
                    lambda: print("  Registration finished."))

    # Execute
    final_transform = reg.Execute(fixed, moving)

    print(f"\n  Final metric value : {reg.GetMetricValue():.6f}")
    print(f"  Optimiser stop cond: {reg.GetOptimizerStopConditionDescription()}")
    print(f"  Final transform:\n    {final_transform}")

    return final_transform


def resample(fixed: sitk.Image, moving: sitk.Image,
             transform: sitk.Transform) -> sitk.Image:
    """Resample *moving* onto the *fixed* grid using *transform*."""
    print("\n── Resampling ──")
    resampled = sitk.Resample(
        moving, fixed, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )
    print("  Done.")
    return resampled

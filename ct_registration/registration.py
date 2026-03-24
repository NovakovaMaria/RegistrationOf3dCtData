"""
Rigid registration of 3-D CT volumes using SimpleITK.
"""

import SimpleITK as sitk


def rigid_register(fixed: sitk.Image, moving: sitk.Image) -> sitk.Transform:
    """
    Rigid (Euler3D) registration with Mattes Mutual Information.

    Parameters
    ----------
    fixed : sitk.Image   – reference volume (before deformation)
    moving : sitk.Image  – volume to align  (after deformation)

    Returns
    -------
    sitk.Transform – the optimised rigid transform
    """
    print("\n── Rigid Registration ──")

    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.15)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=200,
    )
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(fixed, moving)
    print(f"  Final metric value : {reg.GetMetricValue():.6f}")

    return final_transform


def resample(fixed: sitk.Image, moving: sitk.Image,
             transform: sitk.Transform) -> sitk.Image:
    """Resample *moving* onto the *fixed* grid using *transform*."""
    resampled = sitk.Resample(
        moving, fixed, transform,
        sitk.sitkLinear, 0.0, moving.GetPixelID(),
    )
    return resampled

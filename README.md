# Registration of 3D CT Data

Rigid registration of paired 3D CT scans of a sandstone specimen acquired before and after mechanical deformation. The pipeline aligns the post-deformation scan to the pre-deformation reference, then generates quantitative metrics and publication-quality figures.

## Project structure

```
ct_registration/        # Main package
    config.py           # Paths and matplotlib style settings
    io.py               # TIF stack loading and saving
    registration.py     # Rigid registration (SimpleITK)
    masking.py          # Specimen mask generation (Otsu + morphology)
    metrics.py          # MSE, NCC, SSIM computation
    visualization.py    # Quick-preview plots (saved during pipeline)
    figures.py          # Presentation-quality figures (fig1–fig10)
    report.py           # Text report generation
run_registration.py     # Step 1: register and save results
run_visualization.py    # Step 2: generate presentation figures
data/                   # Input TIF stacks (not tracked in git)
results/                # Output figures, registered volume, report (not tracked)
presentation/           # Slides, scripts, mathematical formulation
```

## Data

Place the input TIF stacks (uint16, ~380 × 275 × 275 voxels) in `data/VEC4/VEC4-bin2/`:

| File | Role |
|------|------|
| `VEC4-01-b2.tif` | Fixed image - before deformation |
| `VEC4-02-b2.tif` | Moving image - after deformation |

The `bin2` suffix indicates 2× spatial downsampling from the original scan resolution.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Step 1 - Registration** (run once; output saved to `results/`):

```bash
python run_registration.py
```

This performs rigid registration and saves:
- `results/VEC4-02-b2_registered.tif` - registered volume (float32)
- Quick-preview PNG figures
- `results/registration_report.txt` - metrics and registration settings

**Step 2 - Presentation figures** (requires step 1):

```bash
python run_visualization.py
```

This generates `fig1_overview.png` through `fig10_registration_settings.png` and appends change-map statistics to the report.

## Registration method

| Component | Choice |
|-----------|--------|
| Transform | Rigid - `Euler3DTransform` (3 rotations + 3 translations) |
| Metric | Mattes Mutual Information (50 bins, 15 % random sampling) |
| Optimiser | Regular Step Gradient Descent (max 500 iter / level) |
| Multi-resolution | 3-level pyramid: 4× → 2× → full resolution |
| Initialisation | Centre-of-mass alignment (MOMENTS) |
| Interpolator | Linear (trilinear) |

## Outputs

| File | Description |
|------|-------------|
| `fig1_overview.png` | 3 × 3 grid of central slices (axial / coronal / sagittal) |
| `fig2_misalignment.png` | Magenta/green overlays before and after registration |
| `fig3_difference_maps.png` | Absolute difference maps for all three views |
| `fig4_checkerboard.png` | Checkerboard validation at multiple depths |
| `fig5_metrics.png` | Bar charts and per-slice line plots (MSE, NCC, SSIM) |
| `fig6_histogram.png` | Intensity-difference distributions before/after |
| `fig7_edge_detail.png` | Zoomed ROIs at specimen edges and notch |
| `fig8_summary_dashboard.png` | Single-page summary dashboard |
| `fig9_masked_deformation.png` | Deformation change maps inside specimen mask |
| `fig10_registration_settings.png` | Table of all registration parameters |
| `registration_report.txt` | Quantitative metrics and settings in plain text |

## Requirements

- Python 3.12
- See [requirements.txt](requirements.txt) for pinned dependencies

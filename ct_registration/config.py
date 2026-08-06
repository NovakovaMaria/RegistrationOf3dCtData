"""
Shared configuration: paths, constants, and matplotlib style.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "VEC4", "VEC4-bin2")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FIXED_PATH = os.path.join(DATA_DIR, "VEC4-01-b2.tif")    # Before deformation
MOVING_PATH = os.path.join(DATA_DIR, "VEC4-02-b2.tif")    # After deformation
REGISTERED_PATH = os.path.join(RESULTS_DIR, "VEC4-02-b2_registered.tif")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Colour palette ─────────────────────────────────────────────────────────

TEAL = "#009688"
CORAL = "#FF6F61"
SLATE = "#34495E"

# ─── Matplotlib global style ────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.pad_inches": 0.15,
})

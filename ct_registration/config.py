import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "VEC4", "VEC4-bin2")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

FIXED_PATH = os.path.join(DATA_DIR, "VEC4-01-b2.tif")
MOVING_PATH = os.path.join(DATA_DIR, "VEC4-02-b2.tif")
REGISTERED_PATH = os.path.join(RESULTS_DIR, "VEC4-02-b2_registered.tif")

os.makedirs(RESULTS_DIR, exist_ok=True)

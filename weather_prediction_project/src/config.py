"""Project configuration: paths, target definition, and evaluation periods."""

"""
config.py is the only single source of truth of the whole experiment.
This file configs all "optimal values" here and all other files reads the values here
All different changes can be done here. It prevents complexity of data changes.

"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# relative path here enables project to be run on every different machines.
# do not use absolute path in "root"
DATA_DIR = PROJECT_ROOT / "data"

# input and output files:
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"  # for plots.py

METRIC_DIR = OUTPUT_DIR / "metrics"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
# used by experiment.py
MODEL_DIR = OUTPUT_DIR / "models"  # persisted fitted model and its metadata

# data files being used.
RAW_DATA_PATH = DATA_DIR / "weather_prediction_dataset.csv"
PICNIC_LABELS_PATH = DATA_DIR / "weather_prediction_picnic_labels.csv"
METADATA_PATH = DATA_DIR / "metadata.txt"
MAP_PATH = DATA_DIR / "weather_prediction_dataset_map.png"

# Prediction target of experiment: next-day mean temperature at Basel
TARGET_STATION = "BASEL"
TARGET_VARIABLE = "temp_mean"
TARGET_COLUMN = f"{TARGET_STATION}_{TARGET_VARIABLE}"
FORECAST_HORIZON_DAYS = 1

# Chronological evaluation windows.
# Splits are made on the forecast target date,
# not on the feature date, to avoid bleeding one target value across boundaries.
TRAIN_END = "2006-12-31"
VALIDATION_END = "2008-12-31"
TEST_END = "2010-01-01"

# two global static values:
RANDOM_SEED = 42
# the random seed has to be fixed since tests' MAE has to be the same 
MISSING_SENTINEL = -9999.0

# Physical validity ranges for the source variables.
# Values outside these ranges are impossible or sentinel-like. The cleaning
# step replaces them with causal forward-filled values before feature building.
# A None bound means that side is unbounded.
PHYSICAL_RANGES = {
    "_pressure": (0.9, 1.1),
    "_humidity": (0.0, 1.05),
    "_cloud_cover": (0.0, 8.0),
    "_precipitation": (0.0, None),
    "_sunshine": (0.0, None),
    "_wind_speed": (0.0, None),
    "_wind_gust": (0.0, None),
    "_global_radiation": (0.0, None),
    "_temp_mean": (-40.0, 50.0),
    "_temp_min": (-40.0, 50.0),
    "_temp_max": (-40.0, 50.0),
}
# Physical_Ranges: define the reasonable ranges for variables:
# pressure, humidity... need to be checked if they fit in the ranges

for _directory in (OUTPUT_DIR, FIGURE_DIR, METRIC_DIR, PREDICTION_DIR, MODEL_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
    # add MODEL_DIR to automated dictionary creation list

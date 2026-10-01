"""Feature engineering and chronological split helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

# This file: turn the original observation records to supervised dataset
# divide the dataset by time witout leaking

from .config import (
    FORECAST_HORIZON_DAYS,
    TARGET_COLUMN,
    TRAIN_END,
    VALIDATION_END,
    TEST_END,
)

TEMPERATURE_VARIABLES = ("temp_mean", "temp_min", "temp_max")
DEFAULT_LAGS = (1, 2, 3, 7)
DEFAULT_ROLLING_WINDOWS = (7, 14, 30)
# 1,2,3 are temperary records, 7 is periodic length
# 7,14,30 corrreponding to week, fortnight, month timeslots


def build_supervised_dataset(
    # convert sequence to tables:
    # The original data is a 3654days*163columns time sequence
    # This function turn it to a supervised learning table
    # Every raw is a trainning sample, columns are features, 
    # lables are the second day's temperature

    frame: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build lagged features for next-day prediction.

    All engineered features for row t use information available up to and
    including day t. The target is the station temperature on day t + 1.
    """
    frame = frame.copy().sort_values("date").reset_index(drop=True)
    station = target_column.split("_")[0]
    base_columns = [
        column
        for column in frame.columns
        if column not in {"DATE", "MONTH", "date"}
    ]

    feature_blocks: dict[str, pd.Series] = {
        column: frame[column] for column in base_columns
    }

    # Smooth seasonal encoding. A linear month index would impose a false
    # ordering between December and January.
    # if seasons are encoded with monthly numbers, 12 and 1 are distant

    day_of_year = frame["date"].dt.dayofyear.astype(float)
    angle = 2.0 * np.pi * (day_of_year - 1.0) / 365.25
    feature_blocks["calendar_sin"] = pd.Series(
        np.sin(angle), index=frame.index, name="calendar_sin"
    )
    feature_blocks["calendar_cos"] = pd.Series(
        np.cos(angle), index=frame.index, name="calendar_cos"
    )

    # Local persistence of features and recent history for the target station.
    for variable in TEMPERATURE_VARIABLES:
        column = f"{station}_{variable}"
        if column not in frame.columns:
            continue  # silent skip
        for lag in lags:
            feature_blocks[f"{column}_lag{lag}"] = frame[column].shift(lag)
            # shift(lag): takes data backwards
        for window in rolling_windows:
            feature_blocks[f"{column}_roll{window}"] = frame[column].rolling(
                window, min_periods=window  # for insufficent window: return NaN
            ).mean()
            # rolling(window) is a rolling windowlooking beyond by default
            # timeslot is [t-6,t], containing the target day, but no second day.
            # This design is the key techniche for no leakage

    # Cross-station temperature information captures spatial weather patterns.
    other_stations = [
        # inter-stations features: spacial features
        # capture weathering systems' spacial moving between stations 
        column
        for column in base_columns
        if column.endswith("_temp_mean") and not column.startswith(station)
    ]
    for column in other_stations:
        for lag in (1, 3, 7):
            feature_blocks[f"{column}_lag{lag}"] = frame[column].shift(lag)

    # Slow-moving local drivers provide additional context.
    for variable in ("pressure", "humidity"):
        column = f"{station}_{variable}"
        if column in frame.columns:
            feature_blocks[f"{column}_roll7"] = frame[column].rolling(
                7, min_periods=7
            ).mean()

    features = pd.DataFrame(feature_blocks, index=frame.index)
    target = frame[target_column].shift(-FORECAST_HORIZON_DAYS)
    meta = pd.DataFrame(
        {
            "date": frame["date"], # data ending date
            "target_date": frame["date"].shift(-FORECAST_HORIZON_DAYS), # for division
            "persistence": frame[target_column], # store the predictive value of persistent strong baseline
        }
        # In this design, target[k]==the value of original data: day(k+1)
    )

    valid = (
        target.notna()
        & meta["target_date"].notna()
        & features.notna().all(axis=1)
    )


    features = features.loc[valid].reset_index(drop=True)
    target = target.loc[valid].reset_index(drop=True)
    meta = meta.loc[valid].reset_index(drop=True)
    meta["target_month"] = meta["target_date"].dt.month
    meta["target_doy"] = meta["target_date"].dt.dayofyear
    return features, target, meta


def chronological_split(
    # leakage preventing design 
    meta: pd.DataFrame,
    train_end: str = TRAIN_END,
    validation_end: str = VALIDATION_END,
    test_end: str = TEST_END,
) -> dict[str, np.ndarray]:
    """Return positional indices for train, validation, and test periods."""
    train_end_ts = pd.Timestamp(train_end)
    validation_end_ts = pd.Timestamp(validation_end)
    test_end_ts = pd.Timestamp(test_end)
    target_date = meta["target_date"]
    # make a judgement on labeled date
    """
    The related data are divided into 3 different catagories, train, validation, test.
    train_data: 20000130-20061230 (all for feature dates(aka date))
    validation_data:20061231-20081230
    test_data:20081231-20091231

    (feature) date +1 = labeled date

    """

    masks = {
        "train": target_date <= train_end_ts,
        "validation": (target_date > train_end_ts)
        & (target_date <= validation_end_ts),
        "test": (target_date > validation_end_ts) & (target_date <= test_end_ts),
    }
    return {
        name: np.flatnonzero(mask.to_numpy()) for name, mask in masks.items()
    }
    # The calendar feature uses feature dates, not labeled dates.
    #  


def split_summary(
    # This function write the summary to an split_summary.csv file
    meta: pd.DataFrame, splits: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for name in ("train", "validation", "test"):
        index = splits[name]
        rows.append(
            {
                "split": name,
                "n_samples": int(len(index)),
                "start_target_date": str(meta.loc[index, "target_date"].min().date()),
                "end_target_date": str(meta.loc[index, "target_date"].max().date()),
            }
        )
    return pd.DataFrame(rows)

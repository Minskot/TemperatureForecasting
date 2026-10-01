"""Data loading and dataset-level inspection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import METADATA_PATH, MISSING_SENTINEL, PHYSICAL_RANGES, RAW_DATA_PATH


# data.py: read the csv files and add an usable data column
# test the dataset and keep the data

def load_raw_dataset(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    # Load the tabular weather dataset and add a parsed date column.
    frame = pd.read_csv(path)
    if "DATE" not in frame.columns:
        raise ValueError("Expected a DATE column in the weather dataset.")
    # if file has no "Date" column: return false.


    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["DATE"].astype(str), format="%Y%m%d")
    # uses pandas to resolve DATE to 20001010 format
    frame = frame.sort_values("date").reset_index(drop=True)
    # arrange date in ascending sequence and consequtive.
    return frame


def _range_for_column(column: str) -> tuple[float | None, float | None] | None:
    # Match a column to the physical range rule for its variable suffix.
    for suffix, bounds in PHYSICAL_RANGES.items():
        if column.endswith(suffix):
            return bounds
    return None
    # match acccording to variable's suffix
    # BASEL_temp_mean,OSLO_temp_mean,ROMA_temp_mean: use _temp_mean -> (-40,50)
    # 'None' means this side is boundary-less.
    # return with None means this column has no rule: DATE/MONTH


def _validity_mask(
    series: pd.Series, lower: float | None, upper: float | None
) -> pd.Series:
    # NaN is also treated as invalid so it can be filled in the same pass.
    # this design means absence and out-of-boundary goes to the same repairing path 

    mask = series.notna()
    if lower is not None:
        mask = mask & (series >= lower)
    if upper is not None:
        mask = mask & (series <= upper)
    return mask


def count_physical_range_violations(frame: pd.DataFrame) -> int:
    # Count impossible or sentinel-like values without modifying the frame.
    total = 0
    for column in frame.columns:
        if column in {"DATE", "MONTH", "date"}:
            continue
        bounds = _range_for_column(column)
        if bounds is None:
            continue
        total += int((~_validity_mask(frame[column], *bounds)).sum())
    return total


def _format_rule(lower: float | None, upper: float | None) -> str:
    if lower is not None and upper is not None:
        return f"[{lower}, {upper}]"
    if lower is not None:
        return f">= {lower}"
    if upper is not None:
        return f"<= {upper}"
    return "any"


def clean_physically_invalid_values(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace physically impossible values with causal forward-filled values.

    Only past observations are used for the replacement. If an invalid value
    appears before the first valid observation, the first valid value is used
    as a boundary fallback and the action is recorded in the quality report.
    """
    # First mark outliers as missing;
    # Replace with causal forward filling, using only past data;
    # If the exception occurs at the beginning of the sequence, use the boundary fallback value and record it;
    # Return cleaned data and quality reports 

    cleaned = frame.copy()
    invalid_entries: list[tuple[str, object, float, float | None, float | None, bool]] = []

    for column in cleaned.columns:
        if column in {"DATE", "MONTH", "date"}:
            continue
        bounds = _range_for_column(column)
        if bounds is None:
            continue
        lower, upper = bounds
        valid = _validity_mask(cleaned[column], lower, upper)
        invalid = ~valid
        if not invalid.any(): # clean columns is zero-spend
            continue

        for index in cleaned.index[invalid]:
            invalid_entries.append(
                (
                    column,
                    index,
                    float(cleaned.loc[index, column]),
                    lower,
                    upper,
                    False,
                )
            )
        # if we meet "bad-values": store the original data in invalid_entries

        cleaned.loc[invalid, column] = np.nan
        cleaned[column] = cleaned[column].ffill()
        # set the invalid location to NaN and ffill(): which will skip continuous invalid values 
        remaining = cleaned[column].isna()
        if remaining.any():
            first_valid = cleaned[column].dropna()
            fallback = (
                float(first_valid.iloc[0])
                if not first_valid.empty
                else (0.0 if lower is None else float(lower))
            )
            for index in cleaned.index[remaining]:
                cleaned.loc[index, column] = fallback
                for position, entry in enumerate(invalid_entries):
                    if entry[0] == column and entry[1] == index:
                        invalid_entries[position] = (*entry[:5], True)

    quality_rows = []
    for column, index, value, lower, upper, used_fallback in invalid_entries:
        quality_rows.append(
            # this method creates a quality report:
            {
                "date": cleaned.loc[index, "date"].date().isoformat(),
                "column": column,
                "value": value,
                "rule": _format_rule(lower, upper),
                "action": "boundary_fallback" if used_fallback else "forward_fill",
                "replacement": float(cleaned.loc[index, column]),
            }
        )
    quality_report = pd.DataFrame(
        quality_rows,
        columns=["date", "column", "value", "rule", "action", "replacement"],
    )
    return cleaned, quality_report
    # return cleaned data and quality reports 



def dataset_summary(frame: pd.DataFrame) -> dict:
    # produce a serializable summary of the dataset
    # write into datasset_summary.json
    
    
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"DATE", "MONTH", "date"}
    ]
    stations = sorted(
        {
            column[: -len("_temp_mean")]
            for column in feature_columns
            if column.endswith("_temp_mean")
        }
    )
    numeric_features = frame[feature_columns]

    summary = {
        "n_rows": int(len(frame)),
        "n_feature_columns": int(len(feature_columns)),
        "n_stations": int(len(stations)),
        "stations": stations,
        "start_date": str(frame["date"].min().date()),
        "end_date": str(frame["date"].max().date()),
        "missing_values": int(numeric_features.isna().sum().sum()),
        "sentinel_values": int((numeric_features == MISSING_SENTINEL).sum().sum()),
        "physical_range_violations": count_physical_range_violations(frame),
        "duplicate_dates": int(frame["date"].duplicated().sum()),
    }
    return summary


def save_summary(summary: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def read_metadata_text(path: Path | str = METADATA_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")

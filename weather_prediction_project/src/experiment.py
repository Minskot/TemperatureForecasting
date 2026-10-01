"""End-to-end experiment: tune, train, evaluate, and export results."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import plots
from .config import (
    FORECAST_HORIZON_DAYS,
    METRIC_DIR,
    MODEL_DIR,
    PHYSICAL_RANGES,
    PREDICTION_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_END,
    TRAIN_END,
    VALIDATION_END,
)
from .data import (
    clean_physically_invalid_values,
    dataset_summary,
    load_raw_dataset,
    save_summary,
)
from .features import (
    build_supervised_dataset,
    chronological_split,
    split_summary,
)
from .models import MODEL_FAMILIES

"""
This file stung together data -> features -> models -> plots
Decisions only made on validation sets

"""

TUNING_GRIDS: dict[str, list[dict]] = {
    "ridge": [{"alpha": value} for value in (0.1, 1.0, 10.0, 100.0)],
    "random_forest": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in (None, 10, 16)
        for leaf in (1, 2, 4)
    ],
    "hist_gradient_boosting": [
        {"learning_rate": rate, "max_leaf_nodes": nodes}
        for rate in (0.03, 0.05, 0.1)
        for nodes in (15, 31, 63)
    ],
    "mlp": [
        {"alpha": alpha, "hidden_layer_sizes": layers}
        for alpha in (1e-5, 1e-4, 1e-3)
        for layers in ((64,), (128, 64))
    ],
}


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    # This is a helper function. Calculates MAE, RMSE, R_square
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else float("nan"),
    }


def fit_with_warnings(model, features, target) -> int:
    # Fit and count ConvergenceWarning instead of muting it globally.
    # Each fit counts the number of ConvergenceWarnings.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(features, target)
    return sum(
        1
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    )


def persistence_predictions(meta: pd.DataFrame) -> np.ndarray:
    return meta["persistence"].to_numpy(dtype=float)
    # baseline 1:use today's data as tomorrow


def climatology_predictions(
    # baseline 2
    meta_fit: pd.DataFrame, target_fit: pd.Series, meta_eval: pd.DataFrame
) -> np.ndarray:
    """Monthly mean target from the fitting period, with a global fallback."""
    table = pd.DataFrame(
        {
            "month": meta_fit["target_month"].to_numpy(),
            "target": target_fit.to_numpy(),
        }
    )
    monthly_mean = table.groupby("month")["target"].mean()
    fallback = float(target_fit.mean())
    predictions = meta_eval["target_month"].map(monthly_mean).astype(float)
    return predictions.fillna(fallback).to_numpy(dtype=float)


def tune_family(
    # return(best_recrd,best_params,records)
    family: str,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_validation: pd.DataFrame,
    target_validation: pd.Series,
) -> tuple[dict, dict, pd.DataFrame]:
    records: list[dict] = []
    best_record: dict | None = None
    best_params: dict | None = None

    for params in TUNING_GRIDS[family]:
        model = MODEL_FAMILIES[family](**params)
        # fit() only on train data
        convergence_warnings = fit_with_warnings(
            model, features_train, target_train
        )
        metrics = regression_metrics(
            target_validation, model.predict(features_validation)
        )
        record = {
            "family": family,
            **{key: str(value) for key, value in params.items()},
            **metrics,
            "convergence_warnings": convergence_warnings,
        }
        records.append(record)
        if best_record is None or record["mae"] < best_record["mae"]:
            best_record = record
            best_params = params

    if best_record is None or best_params is None:
        raise RuntimeError(f"No tuning configuration available for family {family}.")
    return best_record, best_params, pd.DataFrame(records)


def rolling_origin_evaluation(
    features: pd.DataFrame,
    target: pd.Series,
    meta: pd.DataFrame,
    family: str,
    params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Expanding-window backtest: retrain before each month of 2009."""
    fold_starts = pd.date_range("2009-01-01", "2010-02-01", freq="MS")
    fold_records: list[dict] = []
    fold_predictions: list[pd.DataFrame] = []

    for index in range(len(fold_starts) - 1):
        start = fold_starts[index]
        end = fold_starts[index + 1]
        train_mask = meta["target_date"] < start
        # expand the window
        test_mask = (meta["target_date"] >= start) & (meta["target_date"] < end)
        if int(test_mask.sum()) == 0:
            continue

        model = MODEL_FAMILIES[family](**params)
        convergence_warnings = fit_with_warnings(
            model, features.loc[train_mask], target.loc[train_mask]
        )
        predictions = model.predict(features.loc[test_mask])
        actual = target.loc[test_mask]
        persistence = meta.loc[test_mask, "persistence"].to_numpy(dtype=float)

        metrics = regression_metrics(actual, predictions)
        persistence_mae = float(mean_absolute_error(actual, persistence))
        fold_records.append(
            {
                "fold_start": start.date(),
                "fold_end": (end - pd.Timedelta(days=1)).date(),
                "n_samples": int(test_mask.sum()),
                "model": family,
                **metrics,
                "persistence_mae": persistence_mae,
                "skill_score_mae": float(1.0 - metrics["mae"] / persistence_mae),
                "convergence_warnings": convergence_warnings,
            }
        )
        fold_frame = meta.loc[test_mask, ["target_date", "persistence"]].copy()
        fold_frame["actual"] = actual.to_numpy()
        fold_frame["rolling_model"] = predictions
        fold_predictions.append(fold_frame)

    predictions_frame = pd.concat(fold_predictions, ignore_index=True)
    metrics_frame = pd.DataFrame(fold_records)
    persistence_mae = float(
        mean_absolute_error(
            predictions_frame["actual"], predictions_frame["persistence"]
        )
    )
    overall = {
        "model": family,
        "n_samples": int(len(predictions_frame)),
        **regression_metrics(
            predictions_frame["actual"], predictions_frame["rolling_model"]
        ),
        "persistence_mae": persistence_mae,
    }
    overall["skill_score_mae"] = float(
        1.0 - overall["mae"] / overall["persistence_mae"]
    )
    return metrics_frame, predictions_frame, pd.DataFrame([overall])


def run() -> dict:
    # The main steps
    started = time.time()
    np.random.seed(RANDOM_SEED)

    raw = load_raw_dataset()
    summary = dataset_summary(raw)
    # prepare data for experiment 
    cleaned, quality_report = clean_physically_invalid_values(raw)
    quality_report.to_csv(METRIC_DIR / "data_quality_report.csv", index=False)
    summary["invalid_values_replaced"] = int(len(quality_report))
    summary["physical_range_rules"] = {
        suffix: list(bounds) for suffix, bounds in PHYSICAL_RANGES.items()
    }
    save_summary(summary, METRIC_DIR / "dataset_summary.json")

    features, target, meta = build_supervised_dataset(cleaned)
    splits = chronological_split(meta)
    split_table = split_summary(meta, splits)
    split_table.to_csv(METRIC_DIR / "split_summary.csv", index=False)

    features_train = features.iloc[splits["train"]]
    # Uses location index for .iloc
    target_train = target.iloc[splits["train"]]
    features_validation = features.iloc[splits["validation"]]
    target_validation = target.iloc[splits["validation"]]
    features_test = features.iloc[splits["test"]]
    target_test = target.iloc[splits["test"]]

    tuning_frames: list[pd.DataFrame] = []
    best_params: dict[str, dict] = {}
    validation_records: dict[str, dict] = {}
    for family in MODEL_FAMILIES:
        # Adjust parameters by family
        record, params, frame = tune_family(
            family,
            features_train,
            target_train,
            features_validation,
            target_validation,
        )
        tuning_frames.append(frame)
        best_params[family] = params
        validation_records[family] = record

    tuning_results = pd.concat(tuning_frames, ignore_index=True)
    tuning_results.to_csv(METRIC_DIR / "tuning_results.csv", index=False)

    # Validation metrics: all models are fitted on the training period only.
    validation_actual = target_validation.to_numpy(dtype=float)
    persistence_validation = persistence_predictions(meta.iloc[splits["validation"]])
    climatology_validation = climatology_predictions(
        meta.iloc[splits["train"]],
        target_train,
        meta.iloc[splits["validation"]],
    )
    baseline_validation_mae = float(
        mean_absolute_error(validation_actual, persistence_validation)
    )
    # uses baseline and 4 models together to evaluate on the validation set
    # The hyperparameters have been set using the verification set
    
    metric_rows: list[dict] = []
    for name, prediction in (
        ("persistence", persistence_validation),
        ("climatology", climatology_validation),
    ):
        row = regression_metrics(validation_actual, prediction)
        row.update(
            {
                "model": name,
                "split": "validation",
                "skill_score_mae": float(
                    1.0 - row["mae"] / baseline_validation_mae
                ),
                # use persistent MAE to be the denominator for skill score
            }
        )
        metric_rows.append(row)
    for family, record in validation_records.items():
        metric_rows.append(
            {
                "model": family,
                "split": "validation",
                "mae": record["mae"],
                "rmse": record["rmse"],
                "r2": record["r2"],
                "skill_score_mae": float(
                    1.0 - record["mae"] / baseline_validation_mae
                ),
            }
        )

    # Final models are refitted on training plus validation data.
    # Since hyperparameters have been defined using validation set, 
    # In this step, no decisions will be made here. 
    # Final two years' data can be reused to train the data instead.
    fit_index = np.concatenate([splits["train"], splits["validation"]])
    features_fit = features.iloc[fit_index]
    target_fit = target.iloc[fit_index]
    test_actual = target_test.to_numpy(dtype=float)
    persistence_test = persistence_predictions(meta.iloc[splits["test"]])
    climatology_test = climatology_predictions(
        meta.iloc[fit_index], target_fit, meta.iloc[splits["test"]]
    )
    baseline_test_mae = float(mean_absolute_error(test_actual, persistence_test))

    test_predictions: dict[str, np.ndarray] = {
        "persistence": persistence_test,
        "climatology": climatology_test,
    }
    fitted_models: dict[str, object] = {}
    final_fit_warnings: dict[str, int] = {}
    for family, params in best_params.items():
        model = MODEL_FAMILIES[family](**params)
        # use train+val to retrain
        final_fit_warnings[family] = fit_with_warnings(
            model, features_fit, target_fit
        )
        fitted_models[family] = model
        test_predictions[family] = model.predict(features_test)
        # test data only applied for once

    for name, prediction in test_predictions.items():
        row = regression_metrics(test_actual, prediction)
        row.update(
            {
                "model": name,
                "split": "test",
                "skill_score_mae": float(
                    1.0 - row["mae"] / baseline_test_mae
                ),
            }
        )
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    column_order = ["model", "split", "mae", "rmse", "r2", "skill_score_mae"]
    metrics = metrics[column_order].sort_values(["split", "mae"]).reset_index(drop=True)
    metrics.to_csv(METRIC_DIR / "metrics.csv", index=False)

    # Select the model on validation, not on the test set.
    validation_model_table = metrics[
        (metrics["split"] == "validation")  # read the table
        & (metrics["model"].isin(MODEL_FAMILIES))
    ].sort_values("mae")
    # exclude both baselines from selection
    best_family = str(validation_model_table.iloc[0]["model"])
    best_validation_mae = float(validation_model_table.iloc[0]["mae"])
    best_test_mae = float(
        metrics[
            (metrics["split"] == "test") & (metrics["model"] == best_family)
        ]["mae"].iloc[0]
    )

    # Persist the validation-selected model and the metadata needed for inference.
    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(fitted_models[best_family], model_path)
    model_metadata = {
        "model_family": best_family,
        "best_params": {
            key: str(value) for key, value in best_params[best_family].items()
        },
        "target_column": TARGET_COLUMN,
        "forecast_horizon_days": FORECAST_HORIZON_DAYS,
        "feature_columns": list(features.columns),
        "random_seed": RANDOM_SEED,
        "periods": {
            "train_target_end": TRAIN_END,
            "validation_target_end": VALIDATION_END,
            "test_target_end": TEST_END,
        },
        "data_quality": {
            "physical_range_rules": {
                suffix: list(bounds) for suffix, bounds in PHYSICAL_RANGES.items()
            },
            "invalid_values_replaced": int(len(quality_report)),
            "report_path": str(
                (METRIC_DIR / "data_quality_report.csv").relative_to(PROJECT_ROOT)
            ),
        },
        "metrics": {
            "validation_mae": best_validation_mae,
            "test_mae": best_test_mae,
        },
        "convergence_warnings": {
            family: int(record.get("convergence_warnings", 0))
            for family, record in validation_records.items()
        },
        "final_fit_convergence_warnings": final_fit_warnings,
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
    }
    with (MODEL_DIR / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(model_metadata, handle, indent=2)

    prediction_frame = meta.iloc[splits["test"]][
        ["date", "target_date", "persistence"]
    ].reset_index(drop=True)
    prediction_frame["actual"] = test_actual
    for name, prediction in test_predictions.items():
        prediction_frame[name] = prediction
    prediction_frame.to_csv(
        PREDICTION_DIR / "test_predictions.csv", index=False
    )

    # Feature importance for the validation-selected model.
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "importance_mean": np.zeros(len(features.columns)),
            "importance_std": np.zeros(len(features.columns)),
        }
    )
    try:
        importance_result = permutation_importance(
            # use permutation_importance instead of tree's internal feature_importances
            fitted_models[best_family],
            features_test,
            target_test,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": importance_result.importances_mean,
                "importance_std": importance_result.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
    except Exception as error:  # pragma: no cover - defensive fallback
        print(f"Permutation importance failed: {error}")
    importance.to_csv(METRIC_DIR / "feature_importance.csv", index=False)

    rolling_metrics, rolling_predictions, rolling_overall = (
        rolling_origin_evaluation(
            features,
            target,
            meta,
            best_family,
            best_params[best_family],
        )
    )
    rolling_metrics.to_csv(METRIC_DIR / "rolling_origin_metrics.csv", index=False)
    rolling_overall.to_csv(METRIC_DIR / "rolling_origin_overall.csv", index=False)
    rolling_predictions.to_csv(
        PREDICTION_DIR / "rolling_origin_predictions.csv", index=False
    )

    test_metrics = metrics[metrics["split"] == "test"].reset_index(drop=True)
    plots.make_all_figures(
        cleaned,
        test_metrics,
        prediction_frame,
        importance,
        rolling_metrics,
        best_family,
    )

    metadata = {
        "best_family": best_family,
        "best_params": {
            key: str(value) for key, value in best_params[best_family].items()
        },
        "best_validation_mae": best_validation_mae,
        "best_test_mae": best_test_mae,
        "n_features": int(features.shape[1]),
        "n_samples": int(features.shape[0]),
        "runtime_seconds": float(time.time() - started),
        "target_column": TARGET_COLUMN,
        "data_quality": {
            "physical_range_violations": int(summary["physical_range_violations"]),
            "invalid_values_replaced": int(len(quality_report)),
            "report_path": str(
                (METRIC_DIR / "data_quality_report.csv").relative_to(PROJECT_ROOT)
            ),
        },
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
        "convergence_warnings": {
            family: int(record.get("convergence_warnings", 0))
            for family, record in validation_records.items()
        },
        "final_fit_convergence_warnings": final_fit_warnings,
    }
    with (METRIC_DIR / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    # write best_family/best_params/samples_number/features_number/running time
    # write these data into run_metadata.json
    print("\nExperiment complete.")
    print(f"Best model selected on validation: {best_family}")
    print(f"Validation MAE: {best_validation_mae:.3f} C")
    print(f"Test MAE: {best_test_mae:.3f} C")
    print(f"Runtime: {metadata['runtime_seconds']:.1f} seconds")
    return metadata

"""Figure generation for experiment outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

from .config import FIGURE_DIR, MAP_PATH, TARGET_COLUMN


COLORS = {
    "actual": "#111827",
    "persistence": "#9ca3af",
    "climatology": "#f59e0b",
    "ridge": "#60a5fa",
    "random_forest": "#34d399",
    "hist_gradient_boosting": "#14b8a6",
    "mlp": "#a78bfa",
    "highlight": "#0f766e",
    "accent": "#dc2626",
}

PRETTY_NAMES = {
    "persistence": "Persistence",
    "climatology": "Monthly climatology",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
    "mlp": "Neural network (MLP)",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _save(figure: plt.Figure, name: str) -> Path:
    path = FIGURE_DIR / name
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return path


def make_task_setup_figure() -> Path:
    _style()
    figure, axis = plt.subplots(figsize=(9.5, 2.7))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 3)
    axis.axis("off")

    boxes = [
        (0.2, 1.1, 2.5, 1.0, "#dbeafe", "Days t-7 ... t\nWeather observations\n18 European stations"),
        (3.2, 1.1, 2.3, 1.0, "#ccfbf1", "Feature builder\nCurrent values, lags,\nrolling averages"),
        (6.0, 1.1, 3.6, 1.0, "#fef3c7", "Regression model\nPredict Basel mean\ntemperature on day t+1"),
    ]
    for x, y, width, height, color, label in boxes:
        axis.add_patch(
            Rectangle((x, y), width, height, facecolor=color, edgecolor="#334155", linewidth=1.0)
        )
        axis.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=9)

    for start, end in ((2.7, 3.2), (5.5, 6.0)):
        axis.add_patch(
            FancyArrowPatch(
                (start, 1.6),
                (end, 1.6),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.4,
                color="#334155",
            )
        )

    axis.text(0.2, 2.55, "Input window", fontsize=10, color="#1e3a8a", weight="bold")
    axis.text(3.2, 2.55, "Representation", fontsize=10, color="#0f766e", weight="bold")
    axis.text(6.0, 2.55, "Forecast target", fontsize=10, color="#92400e", weight="bold")
    axis.text(5.0, 0.45, "One-day-ahead forecast: use data available through day t, predict day t+1", ha="center", fontsize=9.5)
    return _save(figure, "fig_task_setup.png")


def make_data_overview_figure(raw: pd.DataFrame) -> Path:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    series = raw[TARGET_COLUMN]
    smoothed = series.rolling(30, min_periods=15).mean()

    axes[0].plot(raw["date"], series, color="#cbd5e1", linewidth=0.7, label="Daily value")
    axes[0].plot(raw["date"], smoothed, color="#0f766e", linewidth=1.6, label="30-day mean")
    axes[0].set_title("Basel daily mean temperature, 2000-2009")
    axes[0].set_ylabel("Degrees Celsius")
    axes[0].legend(loc="upper right")

    monthly = raw.assign(month=raw["date"].dt.month).groupby("month")[TARGET_COLUMN].mean()
    axes[1].bar(monthly.index, monthly.values, color="#14b8a6", edgecolor="white")
    axes[1].set_title("Average seasonal cycle")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Degrees Celsius")
    axes[1].set_xticks(range(1, 13))

    figure.tight_layout()
    return _save(figure, "fig_data_overview.png")


def make_model_comparison_figure(metrics_test: pd.DataFrame) -> Path:
    _style()
    table = metrics_test.sort_values("mae", ascending=True).copy()
    labels = [PRETTY_NAMES.get(name, name) for name in table["model"]]
    colors = [COLORS.get(name, "#64748b") for name in table["model"]]

    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    bars = axis.barh(labels, table["mae"], color=colors, edgecolor="white")
    axis.invert_yaxis()
    baseline = float(
        table.loc[table["model"] == "persistence", "mae"].iloc[0]
    )
    axis.axvline(baseline, color="#6b7280", linestyle=":", linewidth=1.3)
    axis.text(
        baseline + 0.03,
        len(labels) - 0.4,
        f"Persistence baseline: {baseline:.2f}",
        color="#4b5563",
        fontsize=9,
    )
    for bar, value in zip(bars, table["mae"]):
        axis.text(value + 0.02, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center", fontsize=9)
    axis.set_xlabel("Mean absolute error on the 2009 test period (degrees Celsius)")
    axis.set_title("Forecast error by model")
    figure.tight_layout()
    return _save(figure, "fig_model_comparison.png")


def make_prediction_figure(predictions: pd.DataFrame, best_model: str) -> Path:
    _style()
    frame = predictions.copy()
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    frame = frame.sort_values("target_date")
    columns = ["actual", "persistence", best_model]
    smoothed = frame[["target_date", *columns]].copy()
    for column in columns:
        smoothed[column] = smoothed[column].rolling(7, min_periods=3).mean()

    zoom_start = pd.Timestamp("2009-06-01")
    zoom_end = pd.Timestamp("2009-07-31")
    zoom = smoothed[smoothed["target_date"].between(zoom_start, zoom_end)]

    figure, axes = plt.subplots(2, 1, figsize=(11.5, 6.4), sharey=True)
    for column in columns:
        label = PRETTY_NAMES.get(column, column)
        style = {
            "actual": {"color": COLORS["actual"], "linewidth": 1.8},
            "persistence": {"color": COLORS["persistence"], "linewidth": 1.2, "linestyle": "--"},
        }.get(column, {"color": COLORS["highlight"], "linewidth": 1.8})
        axes[0].plot(smoothed["target_date"], smoothed[column], label=label, **style)
        axes[1].plot(zoom["target_date"], zoom[column], label=label, **style)

    axes[0].set_title("Seven-day smoothed forecast, full test period")
    axes[1].set_title("Zoom on June-July 2009")
    axes[1].set_xlabel("Target date")
    for axis in axes:
        axis.set_ylabel("Degrees Celsius")
    axes[0].legend(ncol=3, loc="upper right")
    figure.tight_layout()
    return _save(figure, "fig_predictions.png")


def make_scatter_figure(predictions: pd.DataFrame, best_model: str, mae: float) -> Path:
    _style()
    actual = predictions["actual"].to_numpy(dtype=float)
    predicted = predictions[best_model].to_numpy(dtype=float)
    lower = float(np.floor(min(actual.min(), predicted.min()))) - 1
    upper = float(np.ceil(max(actual.max(), predicted.max()))) + 1

    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    axis.scatter(actual, predicted, s=14, alpha=0.55, color=COLORS["highlight"], edgecolor="none")
    axis.plot([lower, upper], [lower, upper], color="#374151", linestyle="--", linewidth=1.2)
    axis.set_xlim(lower, upper)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Observed temperature (degrees Celsius)")
    axis.set_ylabel("Predicted temperature (degrees Celsius)")
    axis.set_title(f"Predicted vs observed, {PRETTY_NAMES.get(best_model, best_model)}")
    axis.text(
        0.04,
        0.94,
        f"MAE = {mae:.2f} C",
        transform=axis.transAxes,
        fontsize=10,
        color="#0f172a",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "boxstyle": "round,pad=0.3"},
    )
    figure.tight_layout()
    return _save(figure, "fig_scatter.png")


def make_monthly_error_figure(predictions: pd.DataFrame, best_model: str) -> Path:
    _style()
    frame = predictions.copy()
    frame["month"] = pd.to_datetime(frame["target_date"]).dt.month
    frame["best_error"] = (frame["actual"] - frame[best_model]).abs()
    frame["persistence_error"] = (frame["actual"] - frame["persistence"]).abs()
    grouped = frame.groupby("month")[["best_error", "persistence_error"]].mean()
    positions = np.arange(len(grouped.index))
    width = 0.38

    figure, axis = plt.subplots(figsize=(8.8, 3.8))
    axis.bar(positions - width / 2, grouped["persistence_error"], width, label="Persistence", color=COLORS["persistence"])
    axis.bar(positions + width / 2, grouped["best_error"], width, label=PRETTY_NAMES.get(best_model, best_model), color=COLORS["highlight"])
    axis.set_xticks(positions)
    axis.set_xticklabels([str(month) for month in grouped.index])
    axis.set_xlabel("Month of 2009")
    axis.set_ylabel("Mean absolute error (C)")
    axis.set_title("Error across seasons")
    axis.legend()
    figure.tight_layout()
    return _save(figure, "fig_error_by_month.png")


def make_feature_importance_figure(importance: pd.DataFrame, top_n: int = 20) -> Path:
    _style()
    table = importance.head(top_n).sort_values("importance_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(9.0, 5.4))
    axis.barh(table["feature"], table["importance_mean"], color="#0ea5e9", edgecolor="white")
    axis.set_xlabel("Increase in MAE after shuffling the feature (C)")
    axis.set_title(f"Top {top_n} features by permutation importance")
    figure.tight_layout()
    return _save(figure, "fig_feature_importance.png")


def make_rolling_origin_figure(rolling_metrics: pd.DataFrame) -> Path:
    _style()
    table = rolling_metrics[rolling_metrics["n_samples"] >= 10].copy()
    table["label"] = pd.to_datetime(table["fold_start"]).dt.strftime("%b")
    positions = np.arange(len(table))
    width = 0.38

    figure, axis = plt.subplots(figsize=(9.2, 3.8))
    axis.bar(positions - width / 2, table["persistence_mae"], width, label="Persistence", color=COLORS["persistence"])
    axis.bar(positions + width / 2, table["mae"], width, label="Rolling model", color=COLORS["highlight"])
    axis.set_xticks(positions)
    axis.set_xticklabels(table["label"])
    axis.set_ylabel("Mean absolute error (C)")
    axis.set_xlabel("Forecast month in 2009")
    axis.set_title("Expanding-window evaluation: retrain each month")
    axis.legend()
    figure.tight_layout()
    return _save(figure, "fig_rolling_origin.png")


def make_station_map() -> Path:
    _style()
    output = FIGURE_DIR / "fig_station_map.png"
    with Image.open(MAP_PATH) as image:
        image = image.convert("RGB")
        ratio = 1200 / image.width
        resized = image.resize((1200, int(image.height * ratio)), Image.LANCZOS)
        resized.save(output, quality=92)
    return output


def make_all_figures(
    raw: pd.DataFrame,
    metrics_test: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    best_model: str,
) -> list[Path]:
    best_mae = float(metrics_test.loc[metrics_test["model"] == best_model, "mae"].iloc[0])
    paths = [
        make_task_setup_figure(),
        make_data_overview_figure(raw),
        make_model_comparison_figure(metrics_test),
        make_prediction_figure(predictions, best_model),
        make_scatter_figure(predictions, best_model, best_mae),
        make_monthly_error_figure(predictions, best_model),
        make_feature_importance_figure(importance),
        make_rolling_origin_figure(rolling_metrics),
        make_station_map(),
    ]
    return paths

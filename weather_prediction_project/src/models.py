"""Model definitions used in the experiment."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_SEED

# This file creates an untrained model.
# Evaluation, file processing steps are done within other parts.


def make_ridge(alpha: float = 1.0) -> Pipeline:
    # "make_ridge" is the function to do linear baseline
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha, random_state=RANDOM_SEED)),
        ]
    )

# Ridge linear punishment models uses "StandardScalar".
# Punishing units should be equally set on each parameter,
# but the feasibility depends on Tthe unit corresponding to the characteristic
# different characteristics have different measurements and significance.
# the introduce of StandScaler does standarization for units


def make_random_forest(
    # use Bagging tree to implement Random forest
    # decrease the square root
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    n_estimators: int = 500,
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def make_hist_gradient_boosting(
    # Boosting tree: do serial error correction
    learning_rate: float = 0.05,
    # relatively low learning rate
    max_leaf_nodes: int = 31,
    max_iter: int = 500,
    l2_regularization: float = 1.0,
    # Strengthening the regularization: prevent overfitting intentionally 
) -> HistGradientBoostingRegressor:
    # 
    return HistGradientBoostingRegressor(
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        max_iter=max_iter,
        l2_regularization=l2_regularization,
        random_state=RANDOM_SEED,
    )
    # By the way, due to the experiment results, gradient-boosting is the optimal model



def make_mlp(
    # MLP: neural network part:
    # smooth non-linearity
    alpha: float = 1e-4,
    hidden_layer_sizes: tuple[int, ...] = (128, 64),
) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    alpha=alpha,
                    activation="relu",
                    solver="adam",
                    max_iter=800,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


MODEL_FAMILIES = {
    "ridge": make_ridge,
    "random_forest": make_random_forest,
    "hist_gradient_boosting": make_hist_gradient_boosting,
    "mlp": make_mlp,
}

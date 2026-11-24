"""Model evaluation helpers."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

CV_SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def classification_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Compute threshold-dependent and threshold-free metrics.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_proba : array-like of predicted probabilities for the positive class
    threshold : decision threshold applied to ``y_proba``
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def cv_summary_from_search(name: str, search) -> dict:
    """Extract mean CV metrics of the refitted candidate of a fitted search."""
    idx = search.best_index_
    results = search.cv_results_
    summary = {"model": name}
    for metric in CV_SCORING:
        summary[f"cv_{metric}"] = results[f"mean_test_{metric}"][idx]
        summary[f"cv_{metric}_std"] = results[f"std_test_{metric}"][idx]
    return summary


def cv_summary_from_scores(name: str, scores: dict) -> dict:
    """Summarise the output of ``sklearn.model_selection.cross_validate``."""
    summary = {"model": name}
    for metric in CV_SCORING:
        values = scores[f"test_{metric}"]
        summary[f"cv_{metric}"] = np.mean(values)
        summary[f"cv_{metric}_std"] = np.std(values)
    return summary


def best_f1_threshold(y_true, y_proba) -> float:
    """Return the probability threshold that maximises F1.

    Must be applied to out-of-fold or validation predictions, never to the
    test set, to avoid optimistic estimates.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # The last precision/recall pair has no associated threshold.
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[np.argmax(f1)])


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Variance Inflation Factor for each column of a numeric matrix.

    VIF_j = 1 / (1 - R^2_j), where R^2_j comes from regressing column j on
    all the other columns plus an intercept (ordinary least squares).
    """
    X = X.astype(float)
    vif = {}
    for column in X.columns:
        y = X[column].to_numpy()
        others = X.drop(columns=column).to_numpy()
        design = np.column_stack([np.ones(len(others)), others])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        residuals = y - design @ coef
        ss_res = residuals @ residuals
        ss_tot = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        vif[column] = np.inf if r_squared >= 1 else 1 / (1 - r_squared)
    return pd.Series(vif, name="VIF").sort_values(ascending=False)

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from salifort.data import clean_data, find_raw_data, load_raw_data, rename_columns
from salifort.evaluation import best_f1_threshold, classification_metrics, compute_vif
from salifort.features import encode_features, split_features_target

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    row_a = [0.38, 0.53, 2, 157, 3, 0, 1, 0, "sales", "low"]
    row_b = [0.80, 0.86, 5, 262, 6, 0, 1, 0, "hr", "medium"]
    row_c = [0.90, 0.70, 4, 200, 2, 1, 0, 1, "IT", "high"]
    columns = [
        "satisfaction_level", "last_evaluation", "number_project",
        "average_montly_hours", "time_spend_company", "Work_accident",
        "left", "promotion_last_5years", "Department", "salary",
    ]
    return pd.DataFrame([row_a, row_b, row_c, row_a], columns=columns)


def test_rename_columns_produces_snake_case(raw_sample):
    renamed = rename_columns(raw_sample)
    assert "average_monthly_hours" in renamed.columns
    assert "tenure" in renamed.columns
    assert all(col == col.lower() for col in renamed.columns)


def test_rename_columns_rejects_unexpected_schema(raw_sample):
    with pytest.raises(ValueError):
        rename_columns(raw_sample.drop(columns="salary"))


def test_clean_data_drops_exact_duplicates(raw_sample):
    cleaned = clean_data(raw_sample)
    assert len(cleaned) == 3
    assert not cleaned.duplicated().any()


def test_encode_features_salary_is_ordinal(raw_sample):
    encoded = encode_features(clean_data(raw_sample))
    assert encoded["salary"].tolist() == [0, 1, 2]


def test_encode_features_avoids_dummy_trap(raw_sample):
    encoded = encode_features(clean_data(raw_sample))
    dept_cols = [c for c in encoded.columns if c.startswith("dept_")]
    # Three departments -> two dummy columns (one reference category).
    assert len(dept_cols) == 2
    assert "department" not in encoded.columns


def test_encode_features_rejects_unknown_salary(raw_sample):
    df = clean_data(raw_sample)
    df.loc[0, "salary"] = "very high"
    with pytest.raises(ValueError):
        encode_features(df)


def test_split_features_target_excludes_target(raw_sample):
    X, y = split_features_target(encode_features(clean_data(raw_sample)))
    assert "left" not in X.columns
    assert len(X) == len(y)


def test_classification_metrics_perfect_predictions():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_best_f1_threshold_separates_classes():
    threshold = best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert 0.4 < threshold <= 0.6


def test_compute_vif_detects_collinearity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    X = pd.DataFrame({"a": a, "b": 2 * a + rng.normal(scale=0.01, size=200),
                      "c": rng.normal(size=200)})
    vif = compute_vif(X)
    assert vif["a"] > 100
    assert vif["c"] < 2


@pytest.mark.skipif(
    not any(RAW_DIR.glob("*.csv")), reason="Raw dataset not downloaded"
)
def test_real_dataset_shape_and_duplicates():
    raw = load_raw_data(find_raw_data(RAW_DIR))
    assert raw.shape == (14999, 10)
    assert raw.isna().sum().sum() == 0
    assert len(clean_data(raw)) == 11991

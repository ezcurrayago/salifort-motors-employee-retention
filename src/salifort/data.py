"""Data loading and cleaning utilities for the Salifort Motors HR dataset."""

from pathlib import Path

import pandas as pd

RAW_FILENAME = "HR_comma_sep.csv"
ALTERNATIVE_FILENAMES = ("HR_capstone_dataset.csv",)

TARGET = "left"

COLUMN_RENAMES = {
    "average_montly_hours": "average_monthly_hours",
    "time_spend_company": "tenure",
    "Work_accident": "work_accident",
    "Department": "department",
    "number_project": "number_of_projects",
}

EXPECTED_COLUMNS = [
    "satisfaction_level",
    "last_evaluation",
    "number_of_projects",
    "average_monthly_hours",
    "tenure",
    "work_accident",
    TARGET,
    "promotion_last_5years",
    "department",
    "salary",
]


def find_raw_data(data_dir: Path) -> Path:
    """Return the path of the raw CSV inside ``data_dir``.

    The Kaggle download is named ``HR_comma_sep.csv``; the course copy is
    named ``HR_capstone_dataset.csv``. Both contain the same 14,999 rows.
    """
    data_dir = Path(data_dir)
    for filename in (RAW_FILENAME, *ALTERNATIVE_FILENAMES):
        candidate = data_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No dataset found in {data_dir.resolve()}. Download it from "
        "https://www.kaggle.com/datasets/mfaisalqureshi/"
        "hr-analytics-and-job-prediction and place it in data/raw/."
    )


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load the raw CSV without any transformation."""
    return pd.read_csv(path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to snake_case and fix misspellings."""
    renamed = df.rename(columns=COLUMN_RENAMES)
    missing = set(EXPECTED_COLUMNS) - set(renamed.columns)
    if missing:
        raise ValueError(f"Unexpected schema, missing columns: {sorted(missing)}")
    return renamed


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and drop exact duplicate rows (first occurrence kept)."""
    return (
        rename_columns(df)
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )

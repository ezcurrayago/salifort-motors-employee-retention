"""Feature encoding for the Salifort Motors turnover models."""

import pandas as pd

from salifort.data import TARGET

SALARY_ORDER = ["low", "medium", "high"]

# Variables that may not be available, or may be measured differently,
# before an employee decides to leave (see notebook, robustness check).
POTENTIALLY_LEAKY_FEATURES = ["satisfaction_level", "last_evaluation"]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical variables.

    * ``salary`` is ordinal and mapped to 0 (low), 1 (medium), 2 (high).
    * ``department`` is nominal and one-hot encoded with ``drop_first=True``
      to avoid perfect multicollinearity (dummy variable trap) in linear
      models. Tree-based models are unaffected by this choice.
    """
    unknown = set(df["salary"].unique()) - set(SALARY_ORDER)
    if unknown:
        raise ValueError(f"Unknown salary categories: {sorted(unknown)}")

    encoded = df.copy()
    encoded["salary"] = (
        pd.Categorical(encoded["salary"], categories=SALARY_ORDER, ordered=True)
        .codes.astype("int8")
    )
    encoded = pd.get_dummies(
        encoded, columns=["department"], prefix="dept", drop_first=True, dtype="int8"
    )
    return encoded


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the feature matrix from the target column."""
    return df.drop(columns=TARGET), df[TARGET]

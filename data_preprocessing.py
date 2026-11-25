
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = [
    "study_hours",
    "attendance_pct",
    "previous_score",
    "practice_tests",
    "final_score"
]


def load_dataset(uploaded_file):

    try:
        df = pd.read_csv(uploaded_file)

    except pd.errors.EmptyDataError:
        raise ValueError("CSV file is empty.")

    except Exception as e:
        raise ValueError(f"Unable to read CSV: {e}")

    if df.empty:
        raise ValueError("Dataset contains no records.")

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    return df


def clean_dataset(df):

    df = df.copy()

    rows_before = len(df)

    df = df[REQUIRED_COLUMNS].copy()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    invalid_ranges = {
        "study_hours": (0, 24),
        "attendance_pct": (0, 100),
        "previous_score": (0, 100),
        "practice_tests": (0, None),
        "final_score": (0, 100)
    }

    for col, (minimum, maximum) in invalid_ranges.items():

        invalid = df[col] < minimum

        if maximum is not None:
            invalid = invalid | (df[col] > maximum)

        df.loc[invalid, col] = np.nan

    duplicates_removed = int(df.duplicated().sum())

    df = df.drop_duplicates()

    target_rows_removed = int(
        df["final_score"].isna().sum()
    )

    df = df.dropna(subset=["final_score"])

    if len(df) < 10:
        raise ValueError(
            "At least 10 valid records are required."
        )

    cleaning_info = {
        "rows_before": rows_before,
        "rows_after": len(df),
        "duplicates_removed": duplicates_removed,
        "target_rows_removed": target_rows_removed
    }

    return df, cleaning_info


def get_dataset_info(df):

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "data_types": df.dtypes,
        "description": df.describe(),
        "missing_values": df.isnull().sum(),
        "duplicates": df.duplicated().sum()
    }

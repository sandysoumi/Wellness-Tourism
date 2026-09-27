"""Stage 1 - Data registration.

Validates the dataset in tourism_project/data/ against the business data
dictionary and prints a registration summary. Three gates run in order:

  Gate 1  every expected column is present
  Gate 2  every column has the expected type (numeric vs categorical)
  Gate 3  the target is binary (0/1) with no missing values

Any failure exits with a non-zero code, which stops the GitHub Actions
workflow before downstream jobs run on bad data.
"""
import hashlib
import sys

import pandas as pd

from config import DATA_PATH, EXPECTED_COLUMNS, EXPECTED_DTYPES, TARGET


def validate(df: pd.DataFrame):
    """Return (errors, warnings) found in the dataset."""
    errors, warnings = [], []

    # Gate 1 - required columns
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing expected columns: {missing}")

    # Gate 2 - column types (catches e.g. Age silently turning into text)
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            continue
        actual = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        if actual != expected:
            errors.append(f"{col}: expected {expected}, found {actual} ({df[col].dtype})")

    # Gate 3 - target sanity
    if TARGET in df.columns:
        if df[TARGET].isna().any():
            errors.append(f"Target '{TARGET}' has missing values.")
        values = set(df[TARGET].dropna().unique())
        if not values <= {0, 1}:
            errors.append(f"Target '{TARGET}' must be 0/1, found {sorted(values)}")

    if df.empty:
        errors.append("Dataset has no rows.")

    # Non-blocking notes (reported, but do not stop the pipeline)
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        warnings.append(f"Extra columns not in the data dictionary (dropped in prep): {extra}")
    if "CustomerID" in df.columns and df["CustomerID"].duplicated().any():
        warnings.append(f"{df['CustomerID'].duplicated().sum()} duplicate CustomerIDs.")
    return errors, warnings


def summarize(df: pd.DataFrame, checksum: str):
    print("=" * 62)
    print("DATASET REGISTRATION SUMMARY")
    print("=" * 62)
    id_cols = [c for c in df.columns if c == "CustomerID" or c.startswith("Unnamed")]
    print(f"File        : {DATA_PATH.name}")
    print(f"MD5         : {checksum}   (version fingerprint)")
    print(f"Rows        : {df.shape[0]:,}")
    print(f"Columns     : {df.shape[1]}")
    print(f"Duplicates  : {df.drop(columns=id_cols).duplicated().sum()} (ignoring ID/index columns)")
    n_missing = int(df.isna().sum().sum())
    print(f"Missing     : {n_missing} values")
    if n_missing:
        print(df.isna().sum()[lambda s: s > 0].to_string())

    dist = df[TARGET].value_counts().sort_index()
    print(f"\nTarget distribution ({TARGET}):")
    for label, count in dist.items():
        print(f"  {label}: {count:,} ({count / len(df):.1%})")

    print("\nCategorical levels:")
    for col in df.select_dtypes(exclude="number").columns:
        print(f"  {col}: {df[col].value_counts().to_dict()}")

    print("\nNumeric ranges:")
    num = df.drop(columns=id_cols).select_dtypes(include="number")
    print(num.describe().T[["min", "mean", "max"]].round(2).to_string())


def main():
    if not DATA_PATH.exists():
        sys.exit(f"VALIDATION FAILED - dataset not found at {DATA_PATH}")

    checksum = hashlib.md5(DATA_PATH.read_bytes()).hexdigest()
    df = pd.read_csv(DATA_PATH)
    errors, warnings = validate(df)

    if errors:
        print("VALIDATION FAILED:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    summarize(df, checksum)
    print("\nValidation:")
    for w in warnings:
        print(f"  NOTE: {w}")
    print(f"  PASSED - all {len(EXPECTED_COLUMNS)} expected columns present with the correct types; "
          "target is binary.")
    print(f"\nDataset registered: {DATA_PATH.relative_to(DATA_PATH.parents[2])}")


if __name__ == "__main__":
    main()

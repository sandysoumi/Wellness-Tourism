"""Stage 2 - Data preparation.

Loads the dataset directly from the repository's data folder, cleans it,
creates a stratified train/test split and saves the four split files to
tourism_project/data/processed/. In GitHub Actions that folder is uploaded
as the "data-splits" workflow artifact for the model-building job.

Imputation and encoding are deliberately NOT done here: they live inside the
model pipeline, so they are learned from training data only (no leakage) and
applied identically when the Streamlit app makes predictions.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_PATH, DROP_COLUMNS, RANDOM_STATE, SPLIT_DIR, TARGET,
                    TEST_SIZE)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning rule and report what changed."""
    df = df.copy()
    start = df.shape

    # 1. Remove unnecessary columns: CSV index artefacts, the ID, and
    #    ProductPitched (a one-to-one duplicate of Designation).
    index_cols = [c for c in df.columns if c.startswith("Unnamed")]
    to_drop = [c for c in index_cols + DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=to_drop)
    print(f"Dropped columns     : {to_drop}")

    # 2. Trim stray whitespace in text columns.
    for col in df.select_dtypes(exclude="number").columns:
        df[col] = df[col].str.strip()

    # 3. Fix the 'Fe Male' data-entry typo. 'Unmarried' is deliberately kept
    #    separate from 'Single': the two groups convert at very different
    #    rates (~24% vs ~36%), so merging them would throw away signal.
    n_fixed = int((df["Gender"] == "Fe Male").sum())
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    print(f"Fixed 'Fe Male'     : {n_fixed} rows -> 'Female'")

    # 4. Remove exact duplicate customer records.
    n_dups = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed duplicates  : {n_dups} rows")

    # 5. Rows without a target cannot be used for training.
    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    print(f"Shape               : {start} -> {df.shape}")
    return df


def main():
    df = clean_data(pd.read_csv(DATA_PATH))

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    # Stratify so both splits keep the same ~19% / 81% purchase balance.
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    Xtrain.to_csv(SPLIT_DIR / "Xtrain.csv", index=False)
    Xtest.to_csv(SPLIT_DIR / "Xtest.csv", index=False)
    ytrain.to_csv(SPLIT_DIR / "ytrain.csv", index=False)
    ytest.to_csv(SPLIT_DIR / "ytest.csv", index=False)

    print(f"\nTrain : {Xtrain.shape}, buyer rate {ytrain.mean():.1%}")
    print(f"Test  : {Xtest.shape}, buyer rate {ytest.mean():.1%}")
    print(f"Saved : {sorted(p.name for p in SPLIT_DIR.glob('*.csv'))} "
          f"-> {SPLIT_DIR.relative_to(SPLIT_DIR.parents[2])}/")


if __name__ == "__main__":
    main()

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA = Path("data/raw/stack-overflow-developer-survey-2025-2/survey_results_public.csv")
INTERIM_OUTPUT = Path("data/interim/so_2025_clean.csv")
PROCESSED_OUTPUT = Path("data/processed/so_2025_model_ready.parquet")
TRAIN_SPLIT_OUTPUT = Path("data/processed/so_2025_train.parquet")
TEST_SPLIT_OUTPUT = Path("data/processed/so_2025_test.parquet")
FEATURE_METADATA = Path("data/processed/so_2025_feature_columns.json")

USE_COLUMNS = [
    "Country",
    "EdLevel",
    "YearsCode",
    "Employment",
    "DevType",
    "ConvertedCompYearly",
    "RemoteWork",
    "Currency",
]

NA_VALUES = ["NA", "Other (please specify):"]

EDUCATION_CATEGORIES = {
    "Less than secondary": [
        "Primary/elementary school",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
    ],
    "Some college": [
        "Some college/university study without earning a degree",
        "Associate degree (A.A., A.S., etc.)",
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)",
    ],
    "Bachelor’s degree": ["Bachelor’s degree (B.A., B.S., B.Eng., etc.)"],
    "Master’s degree": ["Master’s degree (M.A., M.S., M.Eng., MBA, etc.)"],
    "Post-grad": ["Doctoral degree (Ph.D)"],
    "Self-taught/other": [
        "Something else",
        "I prefer not to say",
    ],
}

REMOTE_CATEGORIES = {
    "Remote": "Remote",
    "Hybrid (some in-person, leans heavy to flexibility)": "Hybrid-Flexible",
    "Hybrid (some remote, leans heavy to in-person)": "Hybrid-InPerson",
    "Your choice (very flexible, you can come in when you want or just as needed)": "Hybrid-Choice",
    "In-person": "In-person",
}

EMPLOYMENT_ALLOWED = {
    "Employed",
    "Independent contractor, freelancer, or self-employed",
}


def simplify_edlevel(value: Optional[str]) -> Optional[str]:
    if pd.isna(value):
        return None
    for label, raw_values in EDUCATION_CATEGORIES.items():
        if value in raw_values:
            return label
    return value


def parse_years_code(value: Optional[str]) -> Optional[float]:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip()
    if value.startswith("Less than"):
        return 0.5
    if value.startswith("More than"):
        match = re.search(r"(\d+)", value)
        if match:
            return float(match.group(1))
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_devtype(value: Optional[str]) -> Optional[str]:
    if pd.isna(value):
        return None
    return value.split(";")[0].strip()


def normalize_remote(value: Optional[str]) -> Optional[str]:
    if pd.isna(value):
        return None
    return REMOTE_CATEGORIES.get(value, value)


def load_raw_dataset() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA}")
    df = pd.read_csv(RAW_DATA, usecols=USE_COLUMNS, na_values=NA_VALUES)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[df["Employment"].isin(EMPLOYMENT_ALLOWED)]
    df["EdLevelSimplified"] = df["EdLevel"].apply(simplify_edlevel)
    df["YearsCodeNum"] = df["YearsCode"].apply(parse_years_code)
    df["DevTypePrimary"] = df["DevType"].apply(normalize_devtype)
    df["RemoteCategory"] = df["RemoteWork"].apply(normalize_remote)

    df = df.rename(
        columns={
            "ConvertedCompYearly": "CompYearlyUSD",
        }
    )

    df = df.dropna(
        subset=[
            "Country",
            "CompYearlyUSD",
            "EdLevelSimplified",
            "YearsCodeNum",
            "DevTypePrimary",
        ]
    )

    df = df[(df["CompYearlyUSD"] >= 1_000) & (df["CompYearlyUSD"] <= 600_000)]

    selected_columns = [
        "CompYearlyUSD",
        "Country",
        "EdLevelSimplified",
        "YearsCodeNum",
        "Employment",
        "DevTypePrimary",
        "RemoteCategory",
    ]

    df = df[selected_columns].reset_index(drop=True)
    df["SalaryLog10"] = np.log10(df["CompYearlyUSD"])

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    feature_cols = [
        "Country",
        "EdLevelSimplified",
        "YearsCodeNum",
        "DevTypePrimary",
        "RemoteCategory",
    ]
    cat_cols = [
        "Country",
        "EdLevelSimplified",
        "DevTypePrimary",
        "RemoteCategory",
    ]
    num_cols = ["YearsCodeNum"]
    target = df["CompYearlyUSD"].copy()

    encoded = pd.get_dummies(df[cat_cols], drop_first=True)
    encoded[num_cols] = df[num_cols]
    encoded = encoded.fillna(0.0)

    feature_list = encoded.columns.tolist()
    return encoded, target, feature_list


def split_datasets(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 450,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Create bins for stratified splitting (for regression)
    n_bins = 10
    target_binned = pd.qcut(target, q=n_bins, labels=False, duplicates='drop')
    
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=target_binned,  # Stratify by binned target
    )
    train = X_train.copy()
    test = X_test.copy()
    train["CompYearlyUSD"] = y_train
    test["CompYearlyUSD"] = y_test
    return train, test


def main():
    df_raw = load_raw_dataset()
    df_clean = clean_dataset(df_raw)

    INTERIM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    df_clean.to_csv(INTERIM_OUTPUT, index=False)

    features, target, feature_names = encode_features(df_clean)
    processed = features.copy()
    processed["CompYearlyUSD"] = target

    processed.to_parquet(PROCESSED_OUTPUT, index=False)

    train_split, test_split = split_datasets(features, target)
    TRAIN_SPLIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    train_split.to_parquet(TRAIN_SPLIT_OUTPUT, index=False)
    test_split.to_parquet(TEST_SPLIT_OUTPUT, index=False)

    FEATURE_METADATA.write_text(json.dumps({"feature_columns": feature_names}, indent=2))

    print(
        f"Clean dataset saved to {INTERIM_OUTPUT} ({len(df_clean):,} rows).\\n"
        f"Encoded dataset saved to {PROCESSED_OUTPUT}.\\n"
        f"Train split: {TRAIN_SPLIT_OUTPUT.name}, Test split: {TEST_SPLIT_OUTPUT.name}"
    )


if __name__ == "__main__":
    main()



from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "food_delivery_orders_dataset.csv"
OUTPUT_FILE = BASE_DIR / "cleaned_food_delivery_orders.csv"


def clean_data(input_file=INPUT_FILE, output_file=OUTPUT_FILE):
    df = pd.read_csv(input_file)
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Removed duplicates: {before - len(df)}")
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    for col in df.columns:
        if df[col].dtype == "object" or str(df[col].dtype) == "string":
            converted = pd.to_numeric(
                df[col].astype(str).str.replace(",", "", regex=False),
                errors="coerce"
            )
            non_null_original = df[col].notna().sum()
            if non_null_original > 0 and converted.notna().sum() / non_null_original >= 0.80:
                df[col] = converted
    for col in df.columns:
        if any(word in col for word in ["date", "time", "timestamp"]):
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().sum() > 0:
                df[col] = parsed
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")
    for col in numeric_cols:
        if df[col].nunique(dropna=True) > 10:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                df[col] = df[col].clip(lower, upper)

    df.to_csv(output_file, index=False)

    print(f"Cleaned dataset saved to: {output_file}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print("\nMissing values after cleaning:")
    print(df.isna().sum())
    return df


if __name__ == "__main__":
    clean_data()

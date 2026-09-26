from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "cleaned_food_delivery_orders.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def visualize(input_file=INPUT_FILE):
    df = pd.read_csv(input_file)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        plt.figure(figsize=(10, 5))
        missing.plot(kind="bar")
        plt.title("Missing Values by Column")
        plt.xlabel("Column")
        plt.ylabel("Missing Values")
        plt.xticks(rotation=45, ha="right")
        save_plot("01_missing_values.png")
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols[:8]:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        save_plot(f"02_distribution_{col}.png")
    if len(numeric_cols) >= 2:
        plt.figure(figsize=(10, 7))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        save_plot("03_correlation_heatmap.png")
    categorical_cols = df.select_dtypes(include=["object", "string", "category"]).columns
    for col in categorical_cols[:6]:
        counts = df[col].value_counts().head(10)
        if not counts.empty:
            plt.figure(figsize=(9, 5))
            sns.barplot(x=counts.values, y=counts.index)
            plt.title(f"Top Categories - {col}")
            plt.xlabel("Count")
            plt.ylabel(col)
            save_plot(f"04_top_categories_{col}.png")
    for col in numeric_cols[:6]:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        save_plot(f"05_boxplot_{col}.png")

    print(f"Visualizations saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    visualize()

import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_ROOT / "data" / "raw" / "Tweets.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "customer_feedback_real.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("REAL CUSTOMER FEEDBACK DATASET PREPARATION")
print("=" * 70)

print("\nLoading Twitter US Airline Sentiment dataset...")

df = pd.read_csv(RAW_FILE)

print(f"Rows loaded: {len(df)}")
print(f"Columns loaded: {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMN
# ============================================================

if "airline_sentiment" not in df.columns:
    raise ValueError(
        "The expected 'airline_sentiment' column was not found."
    )

if "text" not in df.columns:
    raise ValueError(
        "The expected 'text' column was not found."
    )


# ============================================================
# SELECT RELEVANT COLUMNS
# ============================================================

data = df[
    [
        "text",
        "airline_sentiment"
    ]
].copy()


# ============================================================
# RENAME COLUMNS
# ============================================================

data.rename(
    columns={
        "text": "review",
        "airline_sentiment": "sentiment"
    },
    inplace=True
)


# ============================================================
# CLEAN BASIC DATA
# ============================================================

print("\nRemoving missing reviews...")

data.dropna(
    subset=["review", "sentiment"],
    inplace=True
)


# Remove duplicate reviews

before_duplicates = len(data)

data.drop_duplicates(
    subset=["review"],
    inplace=True
)

after_duplicates = len(data)

print(
    f"Duplicate reviews removed: "
    f"{before_duplicates - after_duplicates}"
)


# ============================================================
# VALIDATE SENTIMENT LABELS
# ============================================================

valid_labels = {
    "positive",
    "neutral",
    "negative"
}

data = data[
    data["sentiment"].isin(valid_labels)
].copy()


# ============================================================
# ADD REVIEW ID
# ============================================================

data.insert(
    0,
    "review_id",
    range(1, len(data) + 1)
)


# ============================================================
# DISPLAY DISTRIBUTION
# ============================================================

print("\nSentiment distribution:")

print(
    data["sentiment"]
    .value_counts()
)


print("\nSentiment percentages:")

print(
    data["sentiment"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)


# ============================================================
# SAVE
# ============================================================

data.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nDataset saved to:")

print(OUTPUT_FILE)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\nFinal dataset shape:")
print(data.shape)

print("\nFirst five rows:")
print(
    data.head().to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("REAL DATASET PREPARATION COMPLETE")
print("=" * 70)
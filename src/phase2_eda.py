import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

RAW_DATA = DATA_DIR / "customer_feedback.csv"
CLEAN_DATA = DATA_DIR / "customer_feedback_cleaned.csv"

REPORT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("PHASE 2 - DATA CLEANING AND EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\n[1] Loading dataset...")

df = pd.read_csv(RAW_DATA)

print(f"Dataset loaded successfully.")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 2. BASIC DATA INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("2. BASIC DATA INSPECTION")
print("=" * 60)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

print("\n" + "=" * 60)
print("3. REMOVING DUPLICATES")
print("=" * 60)

before_duplicates = len(df)

df = df.drop_duplicates()

after_duplicates = len(df)

print(f"Rows before duplicate removal: {before_duplicates}")
print(f"Rows after duplicate removal:  {after_duplicates}")
print(f"Duplicates removed: {before_duplicates - after_duplicates}")


# ============================================================
# 4. HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("4. HANDLING MISSING VALUES")
print("=" * 60)

print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Remove rows where review text or rating is missing
df = df.dropna(subset=["review", "rating"])

# Fill missing sentiment values if necessary
def generate_sentiment(rating):
    if rating >= 4:
        return "positive"
    elif rating == 3:
        return "neutral"
    else:
        return "negative"


df["sentiment"] = df["rating"].apply(generate_sentiment)

print("\nMissing values after cleaning:")
print(df.isnull().sum())


# ============================================================
# 5. CLEAN REVIEW TEXT
# ============================================================

print("\n" + "=" * 60)
print("5. CLEANING REVIEW TEXT")
print("=" * 60)


def clean_text(text):
    """
    Basic NLP text cleaning.
    More advanced NLP preprocessing will be performed in Phase 3.
    """

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


df["cleaned_review"] = df["review"].apply(clean_text)

print("\nOriginal vs cleaned text:")

comparison = df[["review", "cleaned_review"]].head(10)

print(comparison.to_string(index=False))


# ============================================================
# 6. VALIDATE RATINGS
# ============================================================

print("\n" + "=" * 60)
print("6. VALIDATING RATINGS")
print("=" * 60)

print("\nRating distribution:")
print(df["rating"].value_counts().sort_index())

# Keep only valid ratings from 1 to 5
df = df[df["rating"].between(1, 5)]

print("\nValid ratings confirmed: 1 to 5")


# ============================================================
# 7. SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("7. SENTIMENT DISTRIBUTION")
print("=" * 60)

sentiment_counts = df["sentiment"].value_counts()

print(sentiment_counts)

print("\nSentiment percentages:")

sentiment_percentages = (
    df["sentiment"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(sentiment_percentages)


# ============================================================
# 8. RATING STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("8. RATING STATISTICS")
print("=" * 60)

print(f"Average rating: {df['rating'].mean():.2f}")
print(f"Median rating: {df['rating'].median():.2f}")
print(f"Minimum rating: {df['rating'].min()}")
print(f"Maximum rating: {df['rating'].max()}")


# ============================================================
# 9. REVIEW LENGTH ANALYSIS
# ============================================================

df["word_count"] = df["cleaned_review"].apply(
    lambda x: len(x.split())
)

df["character_count"] = df["cleaned_review"].apply(
    len
)

print("\n" + "=" * 60)
print("9. REVIEW LENGTH ANALYSIS")
print("=" * 60)

print(f"Average words per review: {df['word_count'].mean():.2f}")
print(f"Shortest review: {df['word_count'].min()} words")
print(f"Longest review: {df['word_count'].max()} words")


# ============================================================
# 10. VISUALIZATION - SENTIMENT
# ============================================================

print("\n[10] Creating sentiment distribution chart...")

plt.figure(figsize=(8, 5))

sentiment_counts.plot(kind="bar")

plt.title("Customer Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)

plt.tight_layout()

sentiment_chart = SCREENSHOT_DIR / "sentiment_distribution.png"

plt.savefig(sentiment_chart, dpi=300)

plt.close()

print(f"Saved: {sentiment_chart}")


# ============================================================
# 11. VISUALIZATION - RATINGS
# ============================================================

print("\n[11] Creating rating distribution chart...")

plt.figure(figsize=(8, 5))

df["rating"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Customer Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=0)

plt.tight_layout()

rating_chart = SCREENSHOT_DIR / "rating_distribution.png"

plt.savefig(rating_chart, dpi=300)

plt.close()

print(f"Saved: {rating_chart}")


# ============================================================
# 12. VISUALIZATION - REVIEW LENGTH
# ============================================================

print("\n[12] Creating review length chart...")

plt.figure(figsize=(8, 5))

plt.hist(df["word_count"], bins=8)

plt.title("Distribution of Review Length")
plt.xlabel("Number of Words")
plt.ylabel("Number of Reviews")

plt.tight_layout()

length_chart = SCREENSHOT_DIR / "review_length_distribution.png"

plt.savefig(length_chart, dpi=300)

plt.close()

print(f"Saved: {length_chart}")


# ============================================================
# 13. SAVE CLEAN DATASET
# ============================================================

print("\n" + "=" * 60)
print("13. SAVING CLEANED DATASET")
print("=" * 60)

df.to_csv(CLEAN_DATA, index=False)

print(f"Cleaned dataset saved to:")
print(CLEAN_DATA)


# ============================================================
# 14. SAVE EDA SUMMARY
# ============================================================

summary_file = REPORT_DIR / "phase2_eda_summary.txt"

with open(summary_file, "w", encoding="utf-8") as file:

    file.write("AI CAPSTONE PROJECT\n")
    file.write("PHASE 2 - DATA CLEANING AND EDA\n")
    file.write("=" * 60 + "\n\n")

    file.write(f"Total reviews: {len(df)}\n")
    file.write(f"Total columns: {len(df.columns)}\n\n")

    file.write("Missing values:\n")
    file.write(str(df.isnull().sum()))
    file.write("\n\n")

    file.write("Sentiment distribution:\n")
    file.write(str(df["sentiment"].value_counts()))
    file.write("\n\n")

    file.write("Sentiment percentages:\n")
    file.write(str(sentiment_percentages))
    file.write("\n\n")

    file.write(f"Average rating: {df['rating'].mean():.2f}\n")
    file.write(f"Median rating: {df['rating'].median():.2f}\n")
    file.write(f"Minimum rating: {df['rating'].min()}\n")
    file.write(f"Maximum rating: {df['rating'].max()}\n\n")

    file.write(f"Average words per review: {df['word_count'].mean():.2f}\n")
    file.write(f"Shortest review: {df['word_count'].min()}\n")
    file.write(f"Longest review: {df['word_count'].max()}\n")


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PHASE 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated files:")

print(f"1. {CLEAN_DATA}")
print(f"2. {sentiment_chart}")
print(f"3. {rating_chart}")
print(f"4. {length_chart}")
print(f"5. {summary_file}")

print("\nThe cleaned dataset is now ready for Phase 3 NLP processing.")
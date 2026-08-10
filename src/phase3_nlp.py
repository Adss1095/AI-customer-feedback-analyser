import pandas as pd
import re
import html
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"

INPUT_FILE = DATA_DIR / "customer_feedback_real.csv"

OUTPUT_FILE = (
    DATA_DIR / "customer_feedback_real_nlp_ready.csv"
)

VOCAB_FILE = (
    REPORT_DIR / "phase3_real_word_frequency.csv"
)

SUMMARY_FILE = (
    REPORT_DIR / "phase3_real_nlp_summary.txt"
)

REPORT_DIR.mkdir(exist_ok=True)


# ============================================================
# NEGATION WORDS
# ============================================================

NEGATION_WORDS = {
    "no",
    "not",
    "never",
    "neither",
    "nor",
    "cannot",
    "cant",
    "won't",
    "wouldn't",
    "shouldn't",
    "couldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "don't",
    "doesn't",
    "didn't",
    "haven't",
    "hasn't",
    "hadn't"
}


# Keep sentiment-important negation words.
STOP_WORDS = (
    set(ENGLISH_STOP_WORDS)
    - NEGATION_WORDS
)


# ============================================================
# CONTRACTION EXPANSION
# ============================================================

CONTRACTIONS = {

    "can't": "cannot",
    "can't": "cannot",

    "won't": "will not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",

    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",

    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",

    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",

    "i'm": "i am",
    "you're": "you are",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "we're": "we are",
    "they're": "they are",

    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",

    "i'd": "i would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "we'd": "we would",
    "they'd": "they would",

    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",
    "they'll": "they will",

    "there's": "there is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is"
}


# ============================================================
# TEXT PREPROCESSING FUNCTION
# ============================================================

def preprocess_text(text):

    text = str(text)

    # --------------------------------------------------------
    # 1. HTML entity decoding
    # --------------------------------------------------------

    text = html.unescape(text)

    # Example:
    # &amp; -> &
    # &lt;  -> <
    # &gt;  -> >
    

    # --------------------------------------------------------
    # 2. Lowercase
    # --------------------------------------------------------

    text = text.lower()


    # --------------------------------------------------------
    # 3. Remove URLs
    # --------------------------------------------------------

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )


    # --------------------------------------------------------
    # 4. Remove @mentions
    # --------------------------------------------------------

    text = re.sub(
        r"@\w+",
        " ",
        text
    )


    # --------------------------------------------------------
    # 5. Expand contractions
    # --------------------------------------------------------

    for contraction, expansion in CONTRACTIONS.items():

        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            text
        )


    # --------------------------------------------------------
    # 6. Preserve hashtag words
    # --------------------------------------------------------

    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )


    # --------------------------------------------------------
    # 7. Remove punctuation
    # --------------------------------------------------------

    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )


    # --------------------------------------------------------
    # 8. Normalize whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()


    # --------------------------------------------------------
    # 9. Tokenize
    # --------------------------------------------------------

    tokens = text.split()


    # --------------------------------------------------------
    # 10. Remove stop words
    # --------------------------------------------------------

    tokens = [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]


    return " ".join(tokens)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("PHASE 3 - FINAL NLP PREPROCESSING")
print("=" * 70)

print("\n[1] Loading real customer feedback dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# VALIDATE COLUMNS
# ============================================================

required_columns = [
    "review_id",
    "review",
    "sentiment"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns verified.")


# ============================================================
# MISSING VALUES
# ============================================================

print("\n[2] Checking missing values...")

print(
    df[
        ["review", "sentiment"]
    ].isnull().sum()
)

df = df.dropna(
    subset=[
        "review",
        "sentiment"
    ]
).copy()

print(
    f"Rows after missing-value removal: {len(df)}"
)


# ============================================================
# APPLY NLP PREPROCESSING
# ============================================================

print("\n[3] Applying NLP preprocessing...")

df["nlp_ready_review"] = (
    df["review"]
    .apply(preprocess_text)
)


# ============================================================
# REMOVE EMPTY REVIEWS
# ============================================================

before_empty = len(df)

df = df[
    df["nlp_ready_review"]
    .str.strip()
    != ""
].copy()

empty_removed = (
    before_empty - len(df)
)

print(
    f"Empty reviews removed: {empty_removed}"
)


# ============================================================
# SHOW EXAMPLES
# ============================================================

print("\n" + "=" * 70)
print("4. ORIGINAL VS PROCESSED TEXT")
print("=" * 70)

comparison = df[
    [
        "review",
        "nlp_ready_review",
        "sentiment"
    ]
].head(15)

print(
    comparison.to_string(
        index=False
    )
)


# ============================================================
# NEGATION TEST
# ============================================================

print("\n" + "=" * 70)
print("5. NEGATION HANDLING TEST")
print("=" * 70)

test_sentences = [

    "I am not happy with this service",

    "This is not good",

    "I never want to use this airline again",

    "No help from customer support",

    "I didn't like the service",

    "The flight wasn't good",

    "I can't recommend this airline",

    "It's not worth the money"
]

for sentence in test_sentences:

    processed = preprocess_text(
        sentence
    )

    print(
        f"\nOriginal:  {sentence}"
    )

    print(
        f"Processed: {processed}"
    )


# ============================================================
# TOKEN ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("6. TOKEN ANALYSIS")
print("=" * 70)

df["token_count"] = (
    df["nlp_ready_review"]
    .apply(
        lambda text:
        len(text.split())
    )
)

print(
    f"Average tokens per review: "
    f"{df['token_count'].mean():.2f}"
)

print(
    f"Minimum tokens: "
    f"{df['token_count'].min()}"
)

print(
    f"Maximum tokens: "
    f"{df['token_count'].max()}"
)


# ============================================================
# WORD FREQUENCY
# ============================================================

print("\n" + "=" * 70)
print("7. WORD FREQUENCY ANALYSIS")
print("=" * 70)

all_tokens = []

for text in df[
    "nlp_ready_review"
]:

    all_tokens.extend(
        text.split()
    )

word_counts = Counter(
    all_tokens
)

word_frequency_df = pd.DataFrame(
    word_counts.most_common(),
    columns=[
        "word",
        "frequency"
    ]
)

word_frequency_df.to_csv(
    VOCAB_FILE,
    index=False
)

print("\nTop 30 words:")

print(
    word_frequency_df
    .head(30)
    .to_string(
        index=False
    )
)


# ============================================================
# SENTIMENT DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("8. SENTIMENT DISTRIBUTION")
print("=" * 70)

sentiment_counts = (
    df["sentiment"]
    .value_counts()
)

print(
    sentiment_counts
)

sentiment_percentages = (
    df["sentiment"]
    .value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)

print("\nPercentages:")

print(
    sentiment_percentages
)


# ============================================================
# SAVE NLP DATASET
# ============================================================

print("\n" + "=" * 70)
print("9. SAVING NLP-READY DATASET")
print("=" * 70)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    f"NLP-ready dataset saved to:\n"
    f"{OUTPUT_FILE}"
)


# ============================================================
# SAVE SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI CAPSTONE PROJECT\n"
    )

    file.write(
        "PHASE 3 - FINAL NLP PREPROCESSING\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Final reviews: {len(df)}\n"
    )

    file.write(
        f"Average tokens: "
        f"{df['token_count'].mean():.2f}\n"
    )

    file.write(
        f"Minimum tokens: "
        f"{df['token_count'].min()}\n"
    )

    file.write(
        f"Maximum tokens: "
        f"{df['token_count'].max()}\n\n"
    )

    file.write(
        "Preprocessing:\n"
    )

    file.write(
        "- HTML entity decoding\n"
    )

    file.write(
        "- Lowercase conversion\n"
    )

    file.write(
        "- URL removal\n"
    )

    file.write(
        "- User mention removal\n"
    )

    file.write(
        "- Contraction expansion\n"
    )

    file.write(
        "- Hashtag processing\n"
    )

    file.write(
        "- Punctuation removal\n"
    )

    file.write(
        "- Whitespace normalization\n"
    )

    file.write(
        "- Tokenization\n"
    )

    file.write(
        "- Stop-word removal\n"
    )

    file.write(
        "- Negation preservation\n\n"
    )

    file.write(
        "Sentiment distribution:\n"
    )

    file.write(
        str(sentiment_counts)
    )

    file.write(
        "\n\nSentiment percentages:\n"
    )

    file.write(
        str(sentiment_percentages)
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("PHASE 3 FINAL VERSION COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {OUTPUT_FILE}"
)

print(
    f"2. {VOCAB_FILE}"
)

print(
    f"3. {SUMMARY_FILE}"
)

print(
    "\nThe cleaned real-world dataset is ready "
    "for TF-IDF and model training."
)
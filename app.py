import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import html

from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Feedback Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "best_sentiment_model.pkl"
)


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
    "wont",
    "wouldnt",
    "shouldnt",
    "couldnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "dont",
    "doesnt",
    "didnt",
    "havent",
    "hasnt",
    "hadnt"
}


STOP_WORDS = (
    set(ENGLISH_STOP_WORDS)
    - NEGATION_WORDS
)


# ============================================================
# CONTRACTION EXPANSION
# ============================================================

CONTRACTIONS = {

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

    "i'll": "i will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "we'll": "we will",

    "there's": "there is",
    "that's": "that is",
    "what's": "what is",
    "who's": "who is"
}


# ============================================================
# NLP PREPROCESSING
# ============================================================

def preprocess_text(text):

    text = str(text)

    # HTML decoding
    text = html.unescape(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():

        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            text
        )

    # Remove hashtag symbol but keep word
    text = re.sub(
        r"#(\w+)",
        r"\1",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenize
    tokens = text.split()

    # Stop-word removal
    tokens = [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]

    return " ".join(tokens)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_FILE.exists():

        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    return joblib.load(
        MODEL_FILE
    )


# ============================================================
# SENTIMENT DISPLAY
# ============================================================

def sentiment_details(sentiment):

    if sentiment == "positive":

        return (
            "😊",
            "Positive",
            "The feedback expresses a positive customer experience."
        )

    elif sentiment == "negative":

        return (
            "😞",
            "Negative",
            "The feedback indicates dissatisfaction or a negative experience."
        )

    else:

        return (
            "😐",
            "Neutral",
            "The feedback appears neutral or does not strongly express positive or negative sentiment."
        )


# ============================================================
# HEADER
# ============================================================

st.title(
    "💬 Customer Feedback Sentiment Analyzer"
)

st.markdown(
    """
### AI-powered sentiment analysis for customer feedback

Enter a customer review below and the trained machine-learning
model will classify it as **Positive, Neutral, or Negative**.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "📊 Model Information"
    )

    st.markdown(
        """
**Model:** Logistic Regression

**Features:** TF-IDF

**N-grams:** Unigrams + Bigrams

**Training dataset:** 14,381 reviews

**Test accuracy:** 77.20%

**Macro F1:** 72.19%

**Weighted F1:** 77.73%
"""
    )

    st.divider()

    st.subheader(
        "Why Macro F1?"
    )

    st.write(
        """
The dataset contains more negative reviews than
neutral and positive reviews. Macro F1 gives each
sentiment class equal importance during evaluation.
"""
    )

    st.divider()

    st.caption(
        "AI Capstone Project — Customer Feedback Sentiment Analysis"
    )


# ============================================================
# MODEL LOADING
# ============================================================

try:

    model = load_model()

except Exception as error:

    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.stop()


# ============================================================
# INPUT AREA
# ============================================================

st.subheader(
    "📝 Enter Customer Feedback"
)

review = st.text_area(
    "Customer review",
    placeholder=(
        "Example: The service was excellent "
        "and the staff were very helpful."
    ),
    height=150
)


analyze_button = st.button(
    "🔍 Analyze Sentiment",
    type="primary",
    use_container_width=True
)


# ============================================================
# ANALYSIS
# ============================================================

if analyze_button:

    if not review.strip():

        st.warning(
            "Please enter a customer review before analyzing."
        )

    else:

        # ----------------------------------------------------
        # NLP preprocessing
        # ----------------------------------------------------

        processed_review = preprocess_text(
            review
        )

        if not processed_review.strip():

            st.warning(
                "The review does not contain enough usable text "
                "for sentiment analysis."
            )

            st.stop()


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            [processed_review]
        )[0]


        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                [processed_review]
            )[0]
        )

        classes = model.classes_


        probability_dict = {
            class_name: probability
            for class_name, probability
            in zip(
                classes,
                probabilities
            )
        }


        confidence = max(
            probabilities
        )


        # ----------------------------------------------------
        # Sentiment information
        # ----------------------------------------------------

        icon, sentiment_name, description = (
            sentiment_details(
                prediction
            )
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Analysis Result"
        )


        col1, col2 = st.columns(
            [1, 1]
        )


        with col1:

            st.metric(
                "Predicted Sentiment",
                f"{icon} {sentiment_name}"
            )


        with col2:

            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )


        st.info(
            description
        )


        # ====================================================
        # PROBABILITY BREAKDOWN
        # ====================================================

        st.subheader(
            "📈 Sentiment Probability"
        )


        probability_col1, probability_col2, probability_col3 = (
            st.columns(3)
        )


        with probability_col1:

            st.metric(
                "Negative",
                f"{probability_dict.get('negative', 0):.1%}"
            )

            st.progress(
                float(
                    probability_dict.get(
                        "negative",
                        0
                    )
                )
            )


        with probability_col2:

            st.metric(
                "Neutral",
                f"{probability_dict.get('neutral', 0):.1%}"
            )

            st.progress(
                float(
                    probability_dict.get(
                        "neutral",
                        0
                    )
                )
            )


        with probability_col3:

            st.metric(
                "Positive",
                f"{probability_dict.get('positive', 0):.1%}"
            )

            st.progress(
                float(
                    probability_dict.get(
                        "positive",
                        0
                    )
                )
            )


        # ====================================================
        # CONFIDENCE INTERPRETATION
        # ====================================================

        st.subheader(
            "🔎 Prediction Confidence"
        )


        if confidence >= 0.80:

            st.success(
                "High confidence prediction."
            )

        elif confidence >= 0.60:

            st.info(
                "Moderate confidence prediction. "
                "The sentiment is reasonably clear, "
                "but some uncertainty remains."
            )

        else:

            st.warning(
                "Low confidence prediction. "
                "This feedback may require human review."
            )


        # ====================================================
        # PROCESSED TEXT
        # ====================================================

        with st.expander(
            "View NLP-preprocessed text"
        ):

            st.write(
                processed_review
            )


# ============================================================
# FOOTER / PROJECT INFORMATION
# ============================================================

st.divider()

st.caption(
    "Built using Python, NLP, TF-IDF, Logistic Regression, "
    "and Streamlit."
)
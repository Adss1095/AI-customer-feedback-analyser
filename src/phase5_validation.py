import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"
PHASE5_REPORT_DIR = REPORT_DIR / "phase5"

SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
PHASE5_SCREENSHOT_DIR = (
    SCREENSHOT_DIR / "phase5"
)

INPUT_FILE = (
    DATA_DIR /
    "customer_feedback_real_nlp_ready.csv"
)

MODEL_FILE = (
    MODEL_DIR /
    "best_sentiment_model.pkl"
)

PHASE4_PREDICTIONS = (
    REPORT_DIR /
    "phase4_test_predictions.csv"
)

RESULTS_FILE = (
    PHASE5_REPORT_DIR /
    "phase5_validation_results.csv"
)

ERROR_FILE = (
    PHASE5_REPORT_DIR /
    "phase5_error_analysis.csv"
)

REAL_WORLD_FILE = (
    PHASE5_REPORT_DIR /
    "phase5_real_world_predictions.csv"
)

SUMMARY_FILE = (
    PHASE5_REPORT_DIR /
    "phase5_validation_summary.txt"
)

CONFUSION_MATRIX_FILE = (
    PHASE5_SCREENSHOT_DIR /
    "phase5_confusion_matrix.png"
)

CONFIDENCE_FILE = (
    PHASE5_SCREENSHOT_DIR /
    "phase5_prediction_confidence.png"
)


# Create directories

PHASE5_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PHASE5_SCREENSHOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "negative",
    "neutral",
    "positive"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("PHASE 5 - MODEL VALIDATION AND ERROR ANALYSIS")
print("=" * 75)


# ============================================================
# 1. LOAD SAVED MODEL
# ============================================================

print("\n[1] Loading saved best model...")

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_FILE}"
    )

model = joblib.load(
    MODEL_FILE
)

print("Saved model loaded successfully.")

print(
    f"Model file:\n{MODEL_FILE}"
)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n[2] Loading NLP-ready dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Dataset rows: {len(df)}"
)

print(
    f"Dataset columns: {len(df.columns)}"
)


# ============================================================
# 3. VALIDATE DATA
# ============================================================

required_columns = [
    "review",
    "nlp_ready_review",
    "sentiment"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


df = df.dropna(
    subset=[
        "review",
        "nlp_ready_review",
        "sentiment"
    ]
).copy()


df = df[
    df["nlp_ready_review"]
    .str.strip()
    != ""
].copy()


print(
    f"Usable reviews: {len(df)}"
)


# ============================================================
# 4. LOAD PHASE 4 TEST PREDICTIONS
# ============================================================

print("\n[3] Loading Phase 4 test predictions...")

if not PHASE4_PREDICTIONS.exists():

    raise FileNotFoundError(
        "Phase 4 test predictions file was not found:\n"
        f"{PHASE4_PREDICTIONS}"
    )

phase4_test = pd.read_csv(
    PHASE4_PREDICTIONS
)

print(
    f"Phase 4 test predictions loaded: "
    f"{len(phase4_test)} rows"
)


# ============================================================
# 5. VALIDATE SAVED MODEL ON PHASE 4 TEST DATA
# ============================================================

print("\n" + "=" * 75)
print("4. INDEPENDENT MODEL VALIDATION")
print("=" * 75)


# The Phase 4 predictions file contains the NLP-ready
# review text used during testing.

X_test = phase4_test["review"]

y_test = phase4_test[
    "actual_sentiment"
]


print(
    "\nGenerating predictions using the saved model..."
)

predictions = model.predict(
    X_test
)


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision_macro = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

f1_weighted = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


print("\nValidation metrics:")

print(
    f"Accuracy:          {accuracy:.4f}"
)

print(
    f"Macro Precision:   {precision_macro:.4f}"
)

print(
    f"Macro Recall:      {recall_macro:.4f}"
)

print(
    f"Macro F1:          {f1_macro:.4f}"
)

print(
    f"Weighted F1:       {f1_weighted:.4f}"
)


# ============================================================
# 7. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

classification_report_text = (
    classification_report(
        y_test,
        predictions,
        labels=CLASS_NAMES,
        zero_division=0
    )
)

print(
    classification_report_text
)


# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

print("\n[5] Creating confusion matrix...")

cm = confusion_matrix(
    y_test,
    predictions,
    labels=CLASS_NAMES
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "Phase 5 - Sentiment Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_FILE,
    dpi=300
)

plt.close()

print(
    f"Saved:\n{CONFUSION_MATRIX_FILE}"
)


# ============================================================
# 9. ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 75)
print("6. ERROR ANALYSIS")
print("=" * 75)


error_analysis = pd.DataFrame({

    "review":
        X_test.values,

    "actual_sentiment":
        y_test.values,

    "predicted_sentiment":
        predictions

})


error_analysis[
    "correct_prediction"
] = (
    error_analysis[
        "actual_sentiment"
    ]
    ==
    error_analysis[
        "predicted_sentiment"
    ]
)


errors_only = error_analysis[
    ~error_analysis[
        "correct_prediction"
    ]
].copy()


print(
    f"\nTotal test predictions: "
    f"{len(error_analysis)}"
)

print(
    f"Correct predictions: "
    f"{error_analysis['correct_prediction'].sum()}"
)

print(
    f"Incorrect predictions: "
    f"{len(errors_only)}"
)

print(
    f"Error rate: "
    f"{len(errors_only) / len(error_analysis):.2%}"
)


# ============================================================
# ERROR TYPES
# ============================================================

if len(errors_only) > 0:

    errors_only[
        "error_type"
    ] = (
        errors_only[
            "actual_sentiment"
        ]
        + " -> "
        + errors_only[
            "predicted_sentiment"
        ]
    )


    print("\nMost common error types:")

    print(
        errors_only[
            "error_type"
        ]
        .value_counts()
    )


# Save error analysis

errors_only.to_csv(
    ERROR_FILE,
    index=False
)

print(
    f"\nError analysis saved:\n{ERROR_FILE}"
)


# ============================================================
# 10. DISPLAY SAMPLE ERRORS
# ============================================================

print("\n" + "=" * 75)
print("7. SAMPLE MISCLASSIFIED REVIEWS")
print("=" * 75)


if len(errors_only) > 0:

    sample_errors = (
        errors_only
        .head(15)
    )

    for index, row in (
        sample_errors.iterrows()
    ):

        print("\nReview:")

        print(
            row["review"]
        )

        print(
            f"Actual:    "
            f"{row['actual_sentiment']}"
        )

        print(
            f"Predicted: "
            f"{row['predicted_sentiment']}"
        )

        print("-" * 60)


# ============================================================
# 11. PREDICTION CONFIDENCE
# ============================================================

print("\n" + "=" * 75)
print("8. PREDICTION CONFIDENCE ANALYSIS")
print("=" * 75)


confidence_values = None


if hasattr(
    model,
    "predict_proba"
):

    probabilities = (
        model.predict_proba(
            X_test
        )
    )

    confidence_values = (
        probabilities.max(
            axis=1
        )
    )

    predicted_indices = (
        probabilities.argmax(
            axis=1
        )
    )

    probability_classes = (
        model.classes_
    )

    confidence_predictions = (
        probability_classes[
            predicted_indices
        ]
    )

    confidence_df = pd.DataFrame({

        "review":
            X_test.values,

        "actual_sentiment":
            y_test.values,

        "predicted_sentiment":
            confidence_predictions,

        "confidence":
            confidence_values
    })


    print(
        f"Average prediction confidence: "
        f"{confidence_values.mean():.4f}"
    )

    print(
        f"Minimum confidence: "
        f"{confidence_values.min():.4f}"
    )

    print(
        f"Maximum confidence: "
        f"{confidence_values.max():.4f}"
    )


    # --------------------------------------------------------
    # Confidence categories
    # --------------------------------------------------------

    high_confidence = (
        confidence_values >= 0.80
    )

    medium_confidence = (
        (confidence_values >= 0.60)
        &
        (confidence_values < 0.80)
    )

    low_confidence = (
        confidence_values < 0.60
    )


    print("\nConfidence categories:")

    print(
        f"High (>=80%): "
        f"{high_confidence.sum()}"
    )

    print(
        f"Medium (60-79%): "
        f"{medium_confidence.sum()}"
    )

    print(
        f"Low (<60%): "
        f"{low_confidence.sum()}"
    )


    # --------------------------------------------------------
    # Confidence plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        confidence_values,
        bins=20
    )

    plt.xlabel(
        "Prediction Confidence"
    )

    plt.ylabel(
        "Number of Reviews"
    )

    plt.title(
        "Model Prediction Confidence Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        CONFIDENCE_FILE,
        dpi=300
    )

    plt.close()

    print(
        f"\nConfidence chart saved:\n"
        f"{CONFIDENCE_FILE}"
    )


else:

    print(
        "The selected model does not support "
        "probability predictions."
    )


# ============================================================
# 12. REAL-WORLD PREDICTION TESTING
# ============================================================

print("\n" + "=" * 75)
print("9. REAL-WORLD CUSTOMER FEEDBACK TESTING")
print("=" * 75)


real_world_reviews = [

    "The flight was delayed for five hours and nobody helped us.",

    "Everything was smooth and the staff were extremely helpful.",

    "The flight was okay, nothing special.",

    "I loved the service and the crew were amazing.",

    "My baggage was lost and customer support was useless.",

    "The flight arrived on time and everything was fine.",

    "Very disappointed with the terrible customer service.",

    "The experience was average and acceptable.",

    "The staff were friendly and the flight was comfortable.",

    "I will never use this airline again."
]


real_predictions = model.predict(
    real_world_reviews
)


real_world_results = pd.DataFrame({

    "review":
        real_world_reviews,

    "predicted_sentiment":
        real_predictions
})


# Add probabilities if available

if hasattr(
    model,
    "predict_proba"
):

    real_probabilities = (
        model.predict_proba(
            real_world_reviews
        )
    )

    class_order = (
        model.classes_
    )


    for index, class_name in enumerate(
        class_order
    ):

        real_world_results[
            f"{class_name}_probability"
        ] = (
            real_probabilities[
                :, index
            ]
        )


    real_world_results[
        "confidence"
    ] = (
        real_probabilities
        .max(axis=1)
    )


print(
    real_world_results.to_string(
        index=False
    )
)


real_world_results.to_csv(
    REAL_WORLD_FILE,
    index=False
)


print(
    f"\nReal-world predictions saved:\n"
    f"{REAL_WORLD_FILE}"
)


# ============================================================
# 13. SAVE VALIDATION RESULTS
# ============================================================

validation_results = pd.DataFrame({

    "Metric": [

        "Accuracy",

        "Macro Precision",

        "Macro Recall",

        "Macro F1",

        "Weighted F1"
    ],

    "Score": [

        accuracy,

        precision_macro,

        recall_macro,

        f1_macro,

        f1_weighted
    ]
})


validation_results.to_csv(
    RESULTS_FILE,
    index=False
)


# ============================================================
# 14. SAVE SUMMARY REPORT
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
        "PHASE 5 - MODEL VALIDATION, "
        "ERROR ANALYSIS AND REAL-WORLD TESTING\n"
    )

    file.write(
        "=" * 75 + "\n\n"
    )


    file.write(
        "MODEL:\n"
    )

    file.write(
        "Logistic Regression with TF-IDF\n\n"
    )


    file.write(
        "VALIDATION METRICS:\n"
    )

    file.write(
        f"Accuracy: {accuracy:.4f}\n"
    )

    file.write(
        f"Macro Precision: "
        f"{precision_macro:.4f}\n"
    )

    file.write(
        f"Macro Recall: "
        f"{recall_macro:.4f}\n"
    )

    file.write(
        f"Macro F1: "
        f"{f1_macro:.4f}\n"
    )

    file.write(
        f"Weighted F1: "
        f"{f1_weighted:.4f}\n\n"
    )


    file.write(
        "CLASSIFICATION REPORT:\n"
    )

    file.write(
        classification_report_text
    )


    file.write(
        "\n\nERROR ANALYSIS:\n"
    )

    file.write(
        f"Total test samples: "
        f"{len(error_analysis)}\n"
    )

    file.write(
        f"Correct predictions: "
        f"{error_analysis['correct_prediction'].sum()}\n"
    )

    file.write(
        f"Incorrect predictions: "
        f"{len(errors_only)}\n"
    )


    if len(errors_only) > 0:

        file.write(
            "\nError types:\n"
        )

        file.write(
            str(
                errors_only[
                    "error_type"
                ]
                .value_counts()
            )
        )


    if confidence_values is not None:

        file.write(
            "\n\nCONFIDENCE ANALYSIS:\n"
        )

        file.write(
            f"Average confidence: "
            f"{confidence_values.mean():.4f}\n"
        )

        file.write(
            f"Minimum confidence: "
            f"{confidence_values.min():.4f}\n"
        )

        file.write(
            f"Maximum confidence: "
            f"{confidence_values.max():.4f}\n"
        )

        file.write(
            f"High confidence predictions: "
            f"{high_confidence.sum()}\n"
        )

        file.write(
            f"Medium confidence predictions: "
            f"{medium_confidence.sum()}\n"
        )

        file.write(
            f"Low confidence predictions: "
            f"{low_confidence.sum()}\n"
        )


    file.write(
        "\n\nREAL-WORLD TESTING:\n"
    )

    file.write(
        real_world_results.to_string(
            index=False
        )
    )


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("PHASE 5 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nGenerated files:")

print(
    f"1. {RESULTS_FILE}"
)

print(
    f"2. {ERROR_FILE}"
)

print(
    f"3. {REAL_WORLD_FILE}"
)

print(
    f"4. {SUMMARY_FILE}"
)

print(
    f"5. {CONFUSION_MATRIX_FILE}"
)

print(
    f"6. {CONFIDENCE_FILE}"
)

print(
    "\nModel validation and real-world testing are complete."
)
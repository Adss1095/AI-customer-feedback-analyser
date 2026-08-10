import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import joblib


# ============================================================
# PATH CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "reports"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
MODEL_DIR = PROJECT_ROOT / "models"

INPUT_FILE = (
    DATA_DIR / "customer_feedback_real_nlp_ready.csv"
)

RESULTS_FILE = (
    REPORT_DIR / "phase4_model_comparison.csv"
)

CV_RESULTS_FILE = (
    REPORT_DIR / "phase4_cross_validation.csv"
)

REPORT_FILE = (
    REPORT_DIR / "phase4_model_training_summary.txt"
)

PREDICTIONS_FILE = (
    REPORT_DIR / "phase4_test_predictions.csv"
)

REPORT_DIR.mkdir(exist_ok=True)
SCREENSHOT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

CLASS_NAMES = [
    "negative",
    "neutral",
    "positive"
]


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 75)
print("PHASE 4 - IMPROVED MACHINE LEARNING MODEL TRAINING")
print("=" * 75)

print("\n[1] Loading NLP-ready dataset...")

df = pd.read_csv(INPUT_FILE)

print("Dataset loaded successfully.")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# 2. VALIDATE DATA
# ============================================================

required_columns = [
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
        f"Missing required columns: {missing_columns}"
    )


# Remove missing values

df = df.dropna(
    subset=[
        "nlp_ready_review",
        "sentiment"
    ]
).copy()


# Remove empty reviews

df = df[
    df["nlp_ready_review"]
    .str.strip()
    != ""
].copy()


X = df["nlp_ready_review"]
y = df["sentiment"]


print("\nFinal samples available:")
print(len(df))


# ============================================================
# 3. CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 75)
print("3. CLASS DISTRIBUTION")
print("=" * 75)

class_counts = y.value_counts()

print("\nCounts:")
print(class_counts)

print("\nPercentages:")

class_percentages = (
    y.value_counts(
        normalize=True
    )
    .mul(100)
    .round(2)
)

print(class_percentages)


# ============================================================
# 4. STRATIFIED TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 75)
print("4. STRATIFIED TRAIN / TEST SPLIT")
print("=" * 75)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)

print("\nTraining distribution:")

print(
    y_train.value_counts()
)

print("\nTesting distribution:")

print(
    y_test.value_counts()
)


# ============================================================
# 5. DEFINE TF-IDF CONFIGURATION
# ============================================================

print("\n" + "=" * 75)
print("5. TF-IDF CONFIGURATION")
print("=" * 75)

print(
    "Using unigram + bigram TF-IDF features."
)

print(
    "TF-IDF is fitted separately inside each training fold "
    "to prevent data leakage."
)


def create_vectorizer():

    return TfidfVectorizer(

        lowercase=True,

        ngram_range=(1, 2),

        min_df=2,

        max_df=0.95,

        sublinear_tf=True,

        max_features=50000
    )


# ============================================================
# 6. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Linear SVM":
        LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Multinomial Naive Bayes":
        MultinomialNB()
}


# ============================================================
# 7. CREATE PIPELINES
# ============================================================

pipelines = {}

for model_name, model in models.items():

    pipelines[model_name] = Pipeline(
        steps=[
            (
                "tfidf",
                create_vectorizer()
            ),
            (
                "model",
                model
            )
        ]
    )


# ============================================================
# 8. 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 75)
print("6. 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 75)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


scoring = {

    "accuracy": "accuracy",

    "precision_macro":
        "precision_macro",

    "recall_macro":
        "recall_macro",

    "f1_macro":
        "f1_macro",

    "f1_weighted":
        "f1_weighted"
}


cv_results = []

for model_name, pipeline in pipelines.items():

    print("\n" + "-" * 75)

    print(
        f"Cross-validating: {model_name}"
    )

    scores = cross_validate(

        pipeline,

        X_train,

        y_train,

        cv=cv,

        scoring=scoring,

        n_jobs=-1
    )


    result = {

        "Model": model_name,

        "CV_Accuracy_Mean":
            scores[
                "test_accuracy"
            ].mean(),

        "CV_Accuracy_STD":
            scores[
                "test_accuracy"
            ].std(),

        "CV_Precision_Macro":
            scores[
                "test_precision_macro"
            ].mean(),

        "CV_Recall_Macro":
            scores[
                "test_recall_macro"
            ].mean(),

        "CV_F1_Macro":
            scores[
                "test_f1_macro"
            ].mean(),

        "CV_F1_Weighted":
            scores[
                "test_f1_weighted"
            ].mean()
    }

    cv_results.append(result)


    print(
        f"CV Accuracy: "
        f"{result['CV_Accuracy_Mean']:.4f}"
    )

    print(
        f"CV Precision Macro: "
        f"{result['CV_Precision_Macro']:.4f}"
    )

    print(
        f"CV Recall Macro: "
        f"{result['CV_Recall_Macro']:.4f}"
    )

    print(
        f"CV Macro F1: "
        f"{result['CV_F1_Macro']:.4f}"
    )

    print(
        f"CV Weighted F1: "
        f"{result['CV_F1_Weighted']:.4f}"
    )


cv_results_df = pd.DataFrame(
    cv_results
)


# ============================================================
# 9. TRAIN FINAL VERSION OF EACH MODEL
# ============================================================

print("\n" + "=" * 75)
print("7. TRAINING MODELS ON TRAINING DATA")
print("=" * 75)

test_results = []

trained_pipelines = {}


for model_name, pipeline in pipelines.items():

    print("\n" + "-" * 75)

    print(
        f"Training: {model_name}"
    )

    pipeline.fit(
        X_train,
        y_train
    )

    trained_pipelines[
        model_name
    ] = pipeline


    # Predict test set

    predictions = pipeline.predict(
        X_test
    )


    # Metrics

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


    print(
        f"Test Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Test Macro Precision: "
        f"{precision_macro:.4f}"
    )

    print(
        f"Test Macro Recall: "
        f"{recall_macro:.4f}"
    )

    print(
        f"Test Macro F1: "
        f"{f1_macro:.4f}"
    )

    print(
        f"Test Weighted F1: "
        f"{f1_weighted:.4f}"
    )


    test_results.append({

        "Model": model_name,

        "Test_Accuracy":
            accuracy,

        "Test_Precision_Macro":
            precision_macro,

        "Test_Recall_Macro":
            recall_macro,

        "Test_F1_Macro":
            f1_macro,

        "Test_F1_Weighted":
            f1_weighted
    })


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=CLASS_NAMES,
            zero_division=0
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

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
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    matrix_file = (
        SCREENSHOT_DIR
        / f"phase4_confusion_matrix_{safe_name}.png"
    )

    plt.savefig(
        matrix_file,
        dpi=300
    )

    plt.close()


    print(
        f"Confusion matrix saved: "
        f"{matrix_file}"
    )


# ============================================================
# 10. TEST RESULTS
# ============================================================

test_results_df = pd.DataFrame(
    test_results
)


test_results_df = (
    test_results_df
    .sort_values(
        by="Test_F1_Macro",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\n" + "=" * 75)
print("8. TEST SET MODEL COMPARISON")
print("=" * 75)

print(
    test_results_df.to_string(
        index=False
    )
)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = (
    test_results_df
    .iloc[0]["Model"]
)


best_model_f1 = (
    test_results_df
    .iloc[0]["Test_F1_Macro"]
)


print("\n" + "=" * 75)
print("9. BEST MODEL")
print("=" * 75)

print(
    f"Best model based on Test Macro F1: "
    f"{best_model_name}"
)

print(
    f"Test Macro F1: "
    f"{best_model_f1:.4f}"
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

print("\n[10] Saving evaluation results...")


test_results_df.to_csv(
    RESULTS_FILE,
    index=False
)


cv_results_df.to_csv(
    CV_RESULTS_FILE,
    index=False
)


print(
    f"Test results saved to:\n"
    f"{RESULTS_FILE}"
)

print(
    f"Cross-validation results saved to:\n"
    f"{CV_RESULTS_FILE}"
)


# ============================================================
# 13. SAVE BEST MODEL
# ============================================================

best_pipeline = (
    trained_pipelines[
        best_model_name
    ]
)


best_model_file = (
    MODEL_DIR / "best_sentiment_model.pkl"
)


joblib.dump(
    best_pipeline,
    best_model_file
)


print(
    f"\nBest model saved to:\n"
    f"{best_model_file}"
)


# ============================================================
# 14. SAVE TEST PREDICTIONS
# ============================================================

best_predictions = best_pipeline.predict(
    X_test
)


prediction_df = pd.DataFrame({

    "review":
        X_test.values,

    "actual_sentiment":
        y_test.values,

    "predicted_sentiment":
        best_predictions

})


prediction_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


print(
    f"Test predictions saved to:\n"
    f"{PREDICTIONS_FILE}"
)


# ============================================================
# 15. SAVE TRAINING REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "AI CAPSTONE PROJECT\n"
    )

    file.write(
        "PHASE 4 - MODEL TRAINING AND EVALUATION\n"
    )

    file.write(
        "=" * 75 + "\n\n"
    )

    file.write(
        f"Total dataset samples: {len(df)}\n"
    )

    file.write(
        f"Training samples: {len(X_train)}\n"
    )

    file.write(
        f"Testing samples: {len(X_test)}\n\n"
    )


    file.write(
        "Class distribution:\n"
    )

    file.write(
        str(class_counts)
    )

    file.write(
        "\n\n"
    )


    file.write(
        "Feature extraction:\n"
    )

    file.write(
        "TF-IDF unigram and bigram features\n"
    )

    file.write(
        "Maximum features: 50,000\n"
    )

    file.write(
        "Minimum document frequency: 2\n\n"
    )


    file.write(
        "Models evaluated:\n"
    )

    file.write(
        "1. Logistic Regression\n"
    )

    file.write(
        "2. Linear SVM\n"
    )

    file.write(
        "3. Multinomial Naive Bayes\n\n"
    )


    file.write(
        "5-Fold Cross-Validation Results:\n\n"
    )

    file.write(
        cv_results_df.to_string(
            index=False
        )
    )

    file.write(
        "\n\n"
    )


    file.write(
        "Test Set Results:\n\n"
    )

    file.write(
        test_results_df.to_string(
            index=False
        )
    )

    file.write(
        "\n\n"
    )


    file.write(
        f"Selected best model: "
        f"{best_model_name}\n"
    )

    file.write(
        f"Test Macro F1: "
        f"{best_model_f1:.4f}\n"
    )

    file.write(
        "\nModel selection criterion: "
        "highest Test Macro F1.\n"
    )

    file.write(
        "\nMacro F1 was prioritized because the "
        "dataset is imbalanced across negative, "
        "neutral, and positive sentiment classes.\n"
    )


# ============================================================
# 16. FINAL STATUS
# ============================================================

print("\n" + "=" * 75)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nGenerated files:")

print(
    f"1. {RESULTS_FILE}"
)

print(
    f"2. {CV_RESULTS_FILE}"
)

print(
    f"3. {REPORT_FILE}"
)

print(
    f"4. {PREDICTIONS_FILE}"
)

print(
    f"5. {best_model_file}"
)

print(
    "\nConfusion matrices were saved in the "
    "screenshots folder."
)

print(
    "\nThe best model is ready for detailed "
    "Phase 5 evaluation."
)
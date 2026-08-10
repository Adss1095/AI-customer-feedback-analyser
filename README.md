# 💬 Customer Feedback Sentiment Analysis
https://ai-customer-feedback-analyser-iqhtw5qzunvfz5q3h9sxhi.streamlit.app/
Author:ADITI V BIDKAR

An end-to-end Artificial Intelligence capstone project that uses Natural Language Processing (NLP) and Machine Learning to automatically classify customer feedback into **Positive, Neutral, or Negative** sentiment.

The project covers the complete AI workflow — from real-world dataset preparation and text preprocessing to model training, evaluation, error analysis, and deployment as an interactive Streamlit web application.

---

## 📌 Project Overview

Customer feedback contains valuable information about customer satisfaction, service quality, and potential problems. However, manually analyzing thousands of reviews is time-consuming.

This project develops an AI-powered **Customer Feedback Sentiment Analyzer** that automatically analyzes textual feedback and predicts its sentiment.

The system uses:

- Natural Language Processing (NLP)
- TF-IDF text feature extraction
- Machine Learning classification
- Logistic Regression
- Model validation and error analysis
- Confidence estimation
- Streamlit for deployment

The final application allows a user to enter a customer review and receive:

- Predicted sentiment
- Prediction confidence
- Negative probability
- Neutral probability
- Positive probability
- NLP-preprocessed text

---

# 🎯 Objectives

The main objectives of this project are:

1. Prepare and clean a real-world customer feedback dataset.
2. Perform exploratory data analysis.
3. Apply NLP preprocessing techniques to textual feedback.
4. Convert text into numerical features using TF-IDF.
5. Train multiple machine learning classification models.
6. Compare model performance using appropriate evaluation metrics.
7. Select the best-performing model.
8. Perform detailed validation and error analysis.
9. Test the trained model on new customer feedback.
10. Develop an interactive Streamlit application.
11. Prepare the project for cloud deployment.

---

# 📊 Dataset

The project uses the **Twitter US Airline Sentiment dataset**, containing customer feedback related to airline experiences.

The original dataset contains:

- **14,640 records**
- **15 columns**

Relevant information was extracted and transformed into a simplified dataset containing:

| Column | Description |
|---|---|
| `review_id` | Unique review identifier |
| `review` | Original customer feedback |
| `sentiment` | Sentiment label |

After removing missing reviews and duplicate reviews, the prepared real-world dataset contains:

**14,427 reviews**

The sentiment distribution is:

| Sentiment | Reviews | Percentage |
|---|---:|---:|
| Negative | 9,072 | 63.07% |
| Neutral | 3,027 | 21.05% |
| Positive | 2,284 | 15.88% |

The NLP preprocessing stage produced **14,381 usable reviews** after removing reviews that became empty during preprocessing.

---

# 🧠 Project Workflow

```text
Real-World Dataset
       ↓
Data Preparation
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
NLP Preprocessing
       ↓
TF-IDF Feature Engineering
       ↓
Train/Test Split
       ↓
Machine Learning Models
       ↓
Cross-Validation
       ↓
Model Comparison
       ↓
Best Model Selection
       ↓
Validation & Error Analysis
       ↓
Real-World Prediction Testing
       ↓
Streamlit Application
       ↓
Deployment

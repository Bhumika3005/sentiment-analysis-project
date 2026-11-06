# Sentiment Analysis for User Feedback

A simple machine learning and Natural Language Processing (NLP) web application that analyzes user reviews and classifies them into **Positive**, **Neutral**, or **Negative** sentiments. Built with Python, Scikit-learn, and Streamlit.

---

## Overview

Understanding customer feedback helps businesses identify user satisfaction and areas for improvement. This project uses machine learning to classify review text into sentiment categories. The trained model is integrated with an easy-to-use Streamlit web interface for real-time predictions.

---

## Features

- **Real-Time Prediction**: Enter text to instantly classify sentiment as Positive, Neutral, or Negative.
- **Confidence Score**: Displays prediction probability along with a visual progress bar.
- **Text Preprocessing**: Cleans text, removes punctuation, filters stopwords, and performs lemmatization using NLTK.
- **TF-IDF Vectorization**: Converts text into numerical features using unigrams and bigrams.
- **Interactive Web UI**: A clean, beginner-friendly interface built with Streamlit.

---

## Technologies Used

- **Python 3.8+**
- **Streamlit** (Web App UI)
- **Scikit-learn** (Logistic Regression, TF-IDF Vectorizer)
- **NLTK** (Text preprocessing and lemmatization)
- **Pandas & NumPy** (Data manipulation)
- **Joblib** (Model saving and loading)

---

## Dataset

The model is trained on a sample of **3,000 reviews** from the Amazon Customer Reviews dataset (`Reviews.csv`).

Sentiment labels were assigned based on review ratings:
- **1–2 Stars**: Negative
- **3 Stars**: Neutral
- **4–5 Stars**: Positive

---

## Model Performance

The model was trained using **Logistic Regression** and evaluated on a test set of **603 reviews**:

- **Accuracy**: **79.77%** (~80%)
- **Weighted Precision**: **74.90%**
- **Weighted Recall**: **79.77%**
- **Weighted F1-Score**: **73.85%**

### Classification Report

| Sentiment | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 0.88 | 0.27 | 0.42 | 102 |
| **Neutral** | 0.00 | 0.00 | 0.00 | 45 |
| **Positive** | 0.79 | 0.99 | 0.88 | 456 |

> **Note on Neutral Sentiment**:
> The model performs well on Positive and Negative sentiments, but **Neutral sentiment prediction needs improvement**. This is primarily because neutral reviews make up a very small portion of the dataset (only 45 samples in the test set) and share vocabulary with both positive and negative feedback.

---

## Project Structure

```text
sentiment-analysis-project/
│
├── app.py                  # Streamlit web application
├── sentiment_analysis.py   # Preprocessing and model training script
├── model.pkl               # Trained Logistic Regression model
├── vectorizer.pkl          # Saved TF-IDF vectorizer
├── Reviews.csv             # Customer reviews dataset
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Bhumika3005/sentiment-analysis-project.git
   cd sentiment-analysis-project
   ```

2. **Create and activate a virtual environment** *(optional but recommended)*:
   - On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Predict Sentiment
- Type or paste any review text into the input box.
- Click **Predict Sentiment** to see the classification and confidence score.

### 3. Retrain the Model *(Optional)*
To retrain the model from scratch:
```bash
python sentiment_analysis.py
```

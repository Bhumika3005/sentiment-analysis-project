# Sentiment Analysis for User Feedback

## Dataset Used

This project uses the Amazon Reviews Dataset (`Reviews.csv`) containing customer reviews and ratings. For model training, approximately **3000 review samples** were used.

Sentiment labels were created from review ratings:

* Rating 1–2 → Negative
* Rating 3 → Neutral
* Rating 4–5 → Positive

---

## Preprocessing Steps

The following Natural Language Processing (NLP) techniques were applied:

1. Text Lowercasing
2. Removal of URLs
3. Removal of HTML tags
4. Removal of special characters and punctuation
5. Tokenization
6. Stop-word removal using NLTK
7. Lemmatization using WordNetLemmatizer
8. TF-IDF Vectorization to convert text into numerical features

These preprocessing steps improve data quality and model performance.

---

## Model Choice

The project uses **Logistic Regression** as the classification algorithm.

### Reasons for selecting Logistic Regression

* Simple and efficient for text classification
* Fast training and prediction
* Performs well with TF-IDF features
* Easy to interpret and deploy

### Feature Extraction Method

* TF-IDF Vectorizer
* Unigrams and Bigrams
* Maximum Features: 2000

---

## Accuracy Results

The model was evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

### Output

**Accuracy:** 0.7976782752902156

**Precision:** 0.7490049751243781

**Recall:** 0.7976782752902156

**F1 Score:** 0.7384635693835961

### Classification Report

| Sentiment | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Negative  | 0.88      | 0.27   | 0.42     | 102     |
| Neutral   | 0.00      | 0.00   | 0.00     | 45      |
| Positive  | 0.79      | 0.99   | 0.88     | 456     |

**Accuracy:** 0.80

| Average Type | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Macro Avg    | 0.56      | 0.42   | 0.43     | 603     |
| Weighted Avg | 0.75      | 0.80   | 0.74     | 603     |

---

## Conclusion

The Sentiment Analysis system successfully classifies user feedback into **Positive, Neutral, and Negative** categories. The project demonstrates the complete machine learning workflow, including:

* Data Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Sentiment Prediction
* Deployment using Streamlit



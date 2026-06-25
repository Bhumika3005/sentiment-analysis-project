import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

print("Loading dataset...")

# Load only 3000 records
df = pd.read_csv("Reviews.csv", nrows=3000)

df = df[['Text', 'Score']]
df.dropna(inplace=True)

# Convert score to sentiment
def get_sentiment(score):
    if score <= 2:
        return "Negative"
    elif score == 3:
        return "Neutral"
    else:
        return "Positive"

df['Sentiment'] = df['Score'].apply(get_sentiment)

# Custom examples
extra_data = pd.DataFrame({
    "Text": [
        "it is okay",
        "it is ok",
        "it is okay but not bad",
        "it is ok but not bad",
        "average product",
        "acceptable quality",
        "works as expected",
        "not bad",
        "good product",
        "excellent product",
        "very bad product",
        "not good",
        "poor quality",
        "it is good but not bad"
    ],
    "Sentiment": [
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Neutral",
        "Positive",
        "Positive",
        "Positive",
        "Negative",
        "Negative",
        "Negative",
        "Positive"
    ]
})

df = pd.concat(
    [df[['Text', 'Sentiment']], extra_data],
    ignore_index=True
)

print("Records used:", len(df))

# Preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

df["Clean_Text"] = df["Text"].apply(preprocess)

# Features
X = df["Clean_Text"]
y = df["Sentiment"]

vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression(max_iter=500)

print("Training model...")

model.fit(X_train, y_train)

print("Training completed")

# Evaluation
y_pred = model.predict(X_test)

print("\nAccuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall   :", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score :", f1_score(y_test, y_pred, average='weighted'))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully")
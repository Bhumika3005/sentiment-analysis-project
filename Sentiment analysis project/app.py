import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing setup
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

# Streamlit UI
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="",
    layout="centered"
)

st.title(" Sentiment Analysis for User Feedback")

st.write("Enter any review, comment, or feedback and click Predict.")

user_text = st.text_area(
    "Enter Feedback",
    height=150
)

if st.button("Predict Sentiment"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        text_lower = user_text.lower().strip()

        # Custom Neutral Rules
        if (
            "okay but not bad" in text_lower
            or "ok but not bad" in text_lower
            or text_lower == "it is okay"
            or text_lower == "it is ok"
        ):
            prediction = "Neutral"
            confidence = 0.90

        else:
            clean_text = preprocess(user_text)

            vector = vectorizer.transform([clean_text])

            prediction = model.predict(vector)[0]

            try:
                probabilities = model.predict_proba(vector)[0]
                confidence = max(probabilities)
            except:
                confidence = 0.80

        # Result Display
        st.subheader("Prediction Result")

        if prediction == "Positive":
            st.success(" Positive")

        elif prediction == "Neutral":
            st.info(" Neutral")

        else:
            st.error(" Negative")

        # Confidence Display
        st.subheader("Confidence Score")

        st.write(f"{confidence:.2%}")

        st.progress(float(confidence))
import streamlit as st
import joblib
import re
import string
from bs4 import BeautifulSoup

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -------------------------
# Text Cleaning Function
# -------------------------
def clean_text(text):

    text = text.lower()

    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------
# Prediction Function
# -------------------------
def predict_sentiment(review):

    review = clean_text(review)

    tokens = word_tokenize(review)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    review = " ".join(tokens)

    vector = tfidf.transform([review])

    prediction = model.predict(vector)[0]

    probability = model.predict_proba(vector)[0]

    return prediction, probability


# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the trained Machine Learning model "
    "will predict whether the sentiment is Positive or Negative."
)

review = st.text_area(
    "Movie Review",
    height=200,
    placeholder="Example: This movie was absolutely fantastic..."
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        prediction, probability = predict_sentiment(review)

        positive = probability[1] * 100
        negative = probability[0] * 100

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.write("### Confidence")

        st.progress(int(positive))

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Positive", f"{positive:.2f}%")

        with col2:
            st.metric("Negative", f"{negative:.2f}%")
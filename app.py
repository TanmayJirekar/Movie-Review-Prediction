import pandas as pd
import numpy as np
import re
import string
from bs4 import BeautifulSoup

# NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Scikit-Learn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib       
# -----------------------------
# Download NLTK Resources
# -----------------------------
print("Downloading NLTK resources...")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

print("NLTK resources are ready!\n")

# -----------------------------
# Load Dataset
# -----------------------------
print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("IMDB Dataset.csv")

print("\nOriginal Dataset")
print(df.head())

print("\nShape :", df.shape)

# -----------------------------
# Data Cleaning
# -----------------------------
print("\nRemoving duplicate records...")
df.drop_duplicates(inplace=True)

print("Removing missing values...")
df.dropna(inplace=True)

print("Resetting index...")
df.reset_index(drop=True, inplace=True)

print("\nShape after cleaning :", df.shape)

# -----------------------------
# Text Cleaning
# -----------------------------
print("\nInitializing Stopwords and Lemmatizer...")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove HTML Tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove Punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove Numbers
    text = re.sub(r'\d+', '', text)

    # Remove Extra Spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

print("\nCleaning review text...")
df["clean_review"] = df["review"].apply(clean_text)

print("Text Cleaning Completed!")

# -----------------------------
# Tokenization
# -----------------------------
print("\nTokenizing Reviews...")

df["tokens"] = df["clean_review"].apply(word_tokenize)

print("Tokenization Completed!")

# -----------------------------
# Stopword Removal
# -----------------------------
print("\nRemoving Stopwords...")

df["tokens"] = df["tokens"].apply(
    lambda words: [word for word in words if word not in stop_words]
)

print("Stopwords Removed!")

# -----------------------------
# Lemmatization
# -----------------------------
print("\nPerforming Lemmatization...")

df["tokens"] = df["tokens"].apply(
    lambda words: [lemmatizer.lemmatize(word) for word in words]
)

print("Lemmatization Completed!")

# -----------------------------
# Convert Tokens to Sentence
# -----------------------------
print("\nJoining Tokens...")

df["processed_review"] = df["tokens"].apply(lambda words: " ".join(words))

print("Joining Completed!")

print("\nSample Processed Reviews")
print(df[["processed_review"]].head())

# -----------------------------
# Label Encoding
# -----------------------------
print("\nEncoding Sentiment Labels...")

encoder = LabelEncoder()

df["sentiment"] = encoder.fit_transform(df["sentiment"])

print("Encoding Completed!")

print("\nSentiment Distribution")
print(df["sentiment"].value_counts())

# -----------------------------
# Feature Extraction
# -----------------------------
print("\nPerforming TF-IDF Feature Extraction...")

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df["processed_review"])

y = df["sentiment"]

print("TF-IDF Feature Extraction Completed!")

print("\nTF-IDF Matrix Shape :", X.shape)

# -----------------------------
# Train Test Split
# -----------------------------
print("\nSplitting Dataset into Training and Testing (80:20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Dataset Split Completed!")

print("\n" + "=" * 60)
print("Train-Test Split Summary")
print("=" * 60)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

print("\nX_train Shape :", X_train.shape)
print("X_test Shape  :", X_test.shape)

print("y_train Shape :", y_train.shape)
print("y_test Shape  :", y_test.shape)
# -----------------------------
# Train Logistic Regression Model
# -----------------------------

print("\n" + "=" * 60)
print("Training Logistic Regression Model...")
print("=" * 60)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Training Completed!")

# -----------------------------
# Model Evaluation
# -----------------------------

print("\nEvaluating Model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("\nModel Saved Successfully!")
# -----------------------------
# Save Cleaned Dataset
# -----------------------------
print("\nSaving Cleaned Dataset...")

df.to_csv("IMDB_Cleaned.csv", index=False)

print("Dataset Saved Successfully as 'IMDB_Cleaned.csv'")

# -----------------------------
# Pipeline Completed
# -----------------------------
print("\n" + "=" * 60)
print("NLP PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n" + "=" * 60)
print("IMDB Sentiment Prediction")
print("=" * 60)

while True:

    review = input("\nEnter a Movie Review (or type 'exit'): ")

    if review.lower() == "exit":
        print("Goodbye!")
        break

    review = clean_text(review)

    tokens = word_tokenize(review)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    review = " ".join(tokens)

    review_vector = tfidf.transform([review])

    prediction = model.predict(review_vector)[0]

    if prediction == 1:
        print("\n😊 Positive Review")
    else:
        print("\n😞 Negative Review")
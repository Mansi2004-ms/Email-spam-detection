import streamlit as st
import joblib
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered"
)

# ---------------------------------------------------
# Download NLTK Data
# ---------------------------------------------------
@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

download_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------
# Load Model & Vectorizer
# ---------------------------------------------------
@st.cache_resource
def load_model():

    base_path = os.path.dirname(__file__)

    model = joblib.load(
        os.path.join(base_path, "spam_detector_model.pkl")
    )

    vectorizer = joblib.load(
        os.path.join(base_path, "tfidf_vectorizer.pkl")
    )

    return model, vectorizer


model, tfidf_vectorizer = load_model()

# ---------------------------------------------------
# Text Preprocessing
# ---------------------------------------------------

def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+', '', text)

    text = re.sub(r'www\S+', '', text)

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)


def lemmatize(text):

    words = text.split()

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)


def preprocess(text):

    text = clean_text(text)

    text = remove_stopwords(text)

    text = lemmatize(text)

    return text

# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

def predict_spam(text):

    processed_text = preprocess(text)

    vector = tfidf_vectorizer.transform([processed_text])

    prediction = model.predict(vector)

    return prediction[0]

# ---------------------------------------------------
# Streamlit UI
# ---------------------------------------------------

st.title("📧 Email Spam Detection")

st.write(
    "This application predicts whether an email is **Spam** or **Not Spam (Ham)** using Machine Learning."
)

email = st.text_area(
    "Enter Email Content",
    placeholder="Paste your email here..."
)

if st.button("Predict"):

    if email.strip() == "":

        st.warning("Please enter an email.")

    else:

        result = predict_spam(email)

        if result == 1:

            st.error("🚫 This email is SPAM.")

        else:

            st.success("✅ This email is NOT SPAM.")

st.markdown("---")

st.subheader("About")

st.info(
    "This application uses Natural Language Processing (NLP), "
    "TF-IDF Vectorization, and a Machine Learning model "
    "to classify emails as Spam or Not Spam."
)

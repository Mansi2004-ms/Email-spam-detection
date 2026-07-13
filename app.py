import streamlit as st
import joblib
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# -------------------------------------------------
# Download NLTK Resources
# -------------------------------------------------
@st.cache_resource
def download_nltk():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download("omw-1.4")

download_nltk()

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():

    base_path = os.path.dirname(__file__)

    model_path = os.path.join(base_path, "spam_detector_model.pkl")
    vectorizer_path = os.path.join(base_path, "tfidf_vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


model, tfidf_vectorizer = load_model()

# -------------------------------------------------
# Text Preprocessing
# -------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def lemmatize(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# -------------------------------------------------
# Prediction Function
# -------------------------------------------------
def predict_spam(text):

    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize(text)

    vector = tfidf_vectorizer.transform([text])

    prediction = model.predict(vector)

    return prediction[0]

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.title("📧 Email Spam Detection")

st.write(
    "Enter the email content below and the machine learning model will predict whether it is **Spam** or **Not Spam (Ham)**."
)

email = st.text_area(
    "Email Content",
    placeholder="Type or paste an email here..."
)

if st.button("Predict"):

    if email.strip() == "":
        st.warning("Please enter an email.")
    else:

        result = predict_spam(email)

        if result == 1:
            st.error("🚫 Prediction: SPAM")
        else:
            st.success("✅ Prediction: NOT SPAM (HAM)")

st.markdown("---")

st.subheader("About")

st.info(
    "This application uses Natural Language Processing (NLP), "
    "TF-IDF Vectorization, and a Machine Learning model "
    "to classify emails as Spam or Not Spam."
)

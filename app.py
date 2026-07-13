import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (only if not already available in the deployment environment)
# These should ideally be downloaded once during deployment setup or included in the docker image
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except nltk.downloader.DownloadError:
    nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the model and vectorizer
model = joblib.load('spam_detector_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Text preprocessing functions (should match the training preprocessing)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords_func(text):
    return ' '.join([word for word in str(text).split() if word not in stop_words])

def tokenize_and_lemmatize(text):
    tokens = nltk.word_tokenize(text)
    return [lemmatizer.lemmatize(word) for word in tokens]

# Streamlit App
st.title("Email Spam Detector")
st.write("Enter an email body below to check if it's spam or not.")

email_input = st.text_area("Email Body")

if st.button("Predict"):
    if email_input:
        # Preprocess the input email
        cleaned_email = clean_text(email_input)
        stopped_email = remove_stopwords_func(cleaned_email)
        lemmatized_tokens = tokenize_and_lemmatize(stopped_email)
        processed_email = ' '.join(lemmatized_tokens)
        
        # Transform using the loaded TF-IDF vectorizer
        email_vectorized = tfidf_vectorizer.transform([processed_email])
        
        # Make prediction
        prediction = model.predict(email_vectorized)
        
        # Display result
        if prediction[0] == 1:
            st.error("This email is likely SPAM!")
        else:
            st.success("This email is likely NOT SPAM.")
    else:
        st.warning("Please enter some text to predict.")

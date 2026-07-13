# 📧 Email Spam Detection using Machine Learning

## 📌 Project Overview

Email Spam Detection is a Machine Learning and Natural Language Processing (NLP) project that classifies emails as **Spam** or **Not Spam (Ham)** based on their content. The application processes email text, extracts meaningful features using TF-IDF Vectorization, and predicts whether the email is legitimate or spam through an interactive Streamlit web application.

---

## 🎯 Objective

The objective of this project is to build a machine learning model capable of automatically detecting spam emails. This helps filter unwanted messages and demonstrates the practical application of NLP in email security and text classification.

---

## ✨ Features

* Classifies emails as **Spam** or **Not Spam (Ham)**
* Cleans and preprocesses email text
* TF-IDF feature extraction
* Machine Learning-based prediction
* Interactive Streamlit user interface
* Fast and accurate email classification

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* NLTK
* Joblib
* Matplotlib

---

## 📂 Project Structure

```text
Email-Spam-Detection/
│── app.py
│── spam_model.pkl
│── tfidf_vectorizer.pkl
│── requirements.txt
│── README.md
│── .gitignore
│── spam.csv
```

---

## 📊 Dataset

This project uses a publicly available **Email Spam Dataset** containing labeled email or message text.

### Target Classes

* 📩 Ham (Not Spam)
* 🚫 Spam

---

## ⚙️ Machine Learning Workflow

1. Data Collection
2. Data Cleaning
3. Text Preprocessing
4. Tokenization
5. Stop Word Removal
6. TF-IDF Vectorization
7. Train-Test Split
8. Model Training
9. Model Evaluation
10. Streamlit Deployment

---

## 🤖 Machine Learning Algorithm

The project can be trained using one or more of the following algorithms:

* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest

---

## 📈 Model Evaluation

Common evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/email-spam-detection.git
```

### Navigate to the Project Folder

```bash
cd email-spam-detection
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 💻 Example

### Input

```text
Congratulations! You have won a free iPhone. Click the link below to claim your prize.
```

### Output

```text
Prediction: Spam 🚫
```

---

### Input

```text
Hi Mansi, your project presentation is scheduled for tomorrow at 10:00 AM.
```

### Output

```text
Prediction: Ham 📩
```

---

## 📈 Future Improvements

* Display prediction confidence score
* Support multiple languages
* Detect phishing emails
* Integrate live email analysis
* Add explainable AI (feature importance)
* Deploy with Docker and cloud platforms

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Natural Language Processing (NLP)
* Text preprocessing
* TF-IDF Vectorization
* Binary Text Classification
* Machine Learning Model Development
* Streamlit Web Application Development
* GitHub Version Control and Deployment

---

## ⚠️ Disclaimer

This project is developed for educational purposes only. The prediction results are based on the trained machine learning model and should not be considered a complete email security solution.

---

## 👩‍💻 Author

**Mansi Suryawanshi**

B.Tech in Electronics and Telecommunication Engineering
Government College of Engineering, Nagpur


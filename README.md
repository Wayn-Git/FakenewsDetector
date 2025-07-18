# 📰 Fake News Detector

A Machine Learning-based web app that detects whether a given news article is **Real** or **Fake**.

## 🔍 Overview

This project uses Natural Language Processing (NLP) and Machine Learning to classify news text as fake or real. It helps combat misinformation by providing a quick way to verify article content.

## 🚀 Features

- Cleaned and preprocessed news dataset  
- Bag of Words with Bigrams (`CountVectorizer`)  
- Multiple ML models trained and tested  
- Final model with high accuracy and generalization  
- Simple web UI (optional if hosted)  

## 📦 Tech Stack

- **Python**
- **scikit-learn**
- **pandas / NumPy**
- **NLP (CountVectorizer)**
- **Machine Learning models**:
  - Logistic Regression
  - Decision Tree
  - Random Forest

## 🧠 How It Works

1. Dataset is cleaned and preprocessed (removal of stopwords, punctuation, etc.)
2. Text is converted into numerical format using `CountVectorizer` (Bigrams)
3. Multiple models are trained and evaluated
4. The best-performing model is used for predictions


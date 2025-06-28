import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from joblib import dump, load
import re
from nltk.corpus import stopwords
import numpy as np

# Load and preprocess data
df = pd.read_csv('Notebooks/CleanedData.csv')
df = df.drop(columns=['Unnamed: 0'], axis=1)

print("Dataset Info:")
print(f"Total samples: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Label percentages:\n{df['label'].value_counts(normalize=True) * 100}")

# Check for data leakage - ensure content is properly cleaned
X = df['content']
y = df['label']

# Split the data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.2
)

print(f"\nTrain set distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test set distribution:\n{pd.Series(y_test).value_counts()}")

# Use TF-IDF with better parameters
vectorizer = TfidfVectorizer(
    max_features=10000,  # Limit features to prevent overfitting
    min_df=2,           # Ignore terms that appear in less than 2 documents
    max_df=0.95,        # Ignore terms that appear in more than 95% of documents
    ngram_range=(1, 2), # Use unigrams and bigrams
    stop_words='english'
)

X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

print(f"\nFeature matrix shape: {X_train_vect.shape}")

# Define models with better parameters
models = {
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10
    ),
    'Random Forest': RandomForestClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1
    ),
    'Logistic Regression': LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced',  # This helps with imbalanced data
        C=1.0
    )
}

# Train and evaluate models
results = {}
best_model = None
best_score = 0
best_name = ""

print("\n" + "="*50)
print("MODEL TRAINING AND EVALUATION")
print("="*50)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_vect, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_vect)
    y_pred_test = model.predict(X_test_vect)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    results[name] = {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'predictions': y_pred_test
    }
    
    print(f"{name} Results:")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")
    print(f"  Overfitting: {train_accuracy - test_accuracy:.4f}")
    
    # Check prediction distribution
    unique, counts = np.unique(y_pred_test, return_counts=True)
    pred_dist = dict(zip(unique, counts))
    print(f"  Predictions distribution: {pred_dist}")
    
    # Update best model
    if test_accuracy > best_score:
        best_score = test_accuracy
        best_model = model
        best_name = name

print(f"\n🏆 BEST MODEL: {best_name} with accuracy: {best_score:.4f}")

# Detailed evaluation of best model
print("\n" + "="*50)
print("DETAILED EVALUATION OF BEST MODEL")
print("="*50)

y_pred_best = best_model.predict(X_test_vect)
print(f"\nClassification Report for {best_name}:")
print(classification_report(y_test, y_pred_best, target_names=['Real', 'Fake']))

# Check if model is predicting only one class
print(f"\nPrediction distribution:")
unique_pred, counts_pred = np.unique(y_pred_best, return_counts=True)
for label, count in zip(unique_pred, counts_pred):
    print(f"  Class {label}: {count} predictions ({count/len(y_pred_best)*100:.1f}%)")

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(best_model, X_test_vect, y_test)
plt.title(f'Confusion Matrix - {best_name}')
plt.show()

# Save the best model and vectorizer
print(f"\nSaving {best_name} model...")
dump(best_model, 'FakeNewsDetector.joblib')
dump(vectorizer, 'Vectorizer.joblib')

# Load stopwords for prediction function
stop_words = set(stopwords.words('english'))

def cleanText(text):
    """Clean text for prediction"""
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def predict_news(text, model=best_model, vectorizer=vectorizer, clean=True):
    """
    Predict if news is fake or real
    
    Args:
        text (str): News article text
        model: Trained model
        vectorizer: Fitted vectorizer
        clean (bool): Whether to clean the text
    
    Returns:
        str: Prediction with confidence
    """
    if clean:
        text = cleanText(text)
    
    # Transform text
    text_vect = vectorizer.transform([text])
    
    # Get prediction and probability
    prediction = model.predict(text_vect)[0]
    probabilities = model.predict_proba(text_vect)[0]
    confidence = probabilities[prediction] * 100
    
    # Format result
    result = "🟥 FAKE" if prediction == 1 else "🟩 REAL"
    
    return f"{result} ({confidence:.2f}% confident)"

# Test the prediction function
print("\n" + "="*50)
print("TESTING PREDICTIONS")
print("="*50)

test_articles = [
    "ISRO successfully launches another PSLV mission for Earth observation.",
    "Drinking Dettol cures all known diseases, says viral WhatsApp message.",
    "Scientists at MIT develop new breakthrough in quantum computing technology.",
    "Local man discovers aliens in his backyard, government covers it up.",
    "Stock markets show steady growth amid positive economic indicators."
]

for article in test_articles:
    prediction = predict_news(article)
    print(f"📰 Article: {article[:60]}...")
    print(f"🤖 Prediction: {prediction}\n")

# Additional diagnostics
print("="*50)
print("DIAGNOSTIC INFORMATION")
print("="*50)

print(f"Feature vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"Model type: {type(best_model).__name__}")

# Check for data issues
print(f"\nOriginal dataset label distribution:")
print(df['label'].value_counts(normalize=True))

# Check if there are any suspicious patterns
print(f"\nSample of actual vs predicted:")
sample_indices = np.random.choice(len(y_test), 10, replace=False)
for i in sample_indices:
    actual = list(y_test)[i]
    predicted = y_pred_best[i]
    status = "✅" if actual == predicted else "❌"
    print(f"{status} Actual: {actual}, Predicted: {predicted}")
import gradio as gr
from joblib import load

# Load model and vectorizer
vecto = load('UI/Vectorizer(2).joblib')
model = load('UI/FakeNewsDetector(2).joblib')

# Prediction function
def predict_fake_news(text):
    vect = vecto.transform([text])
    prob = model.predict_proba(vect)[0]
    label = model.predict(vect)[0]
    confidence = prob[label] * 100
    result = "🟥 FAKE NEWS" if label == 1 else "🟩 REAL NEWS"
    return f"{result} ({confidence:.2f}% confidence)"


# Gradio Interface
title = "📰 Fake News Detector"
description = """
Enter a news article or headline to check if it's fake or real.  
This model was trained on a labeled dataset using Logistic Regression, Random Forest, and Decision Trees.  
**Best model automatically selected based on test accuracy.**
"""

examples = [
    "The government has launched a new education policy this year.",
    "Scientists say chocolate cures COVID-19 instantly!",
    "NASA confirms the existence of water on Mars.",
    "Aliens spotted in Antarctica, Pentagon confirms.",
]

gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter news content here...", label="📝 News Article Content"),
    outputs=gr.Textbox(label="📢 Prediction"),
    title=title,
    description=description,
    examples=examples,
    theme="default"
).launch()

import gradio as gr
from joblib import load

vecto = load("Models/Vectorizer.joblib")
model = load('../Models/FakeNewsDetector.joblib')

def predict_fake_news(text):
    vect = vecto.transform([text])
    prob = model.predict_proba(vect)[0]
    label = model.predict(vect)[0]
    confidence = prob[label] * 100
    result = "🟥 FAKE NEWS" if label == 1 else "🟩 REAL NEWS"
    return f"{result} ({confidence:.2f}% confidence)"

title = "📰 Fake News Detector"

gr.Interface(
    fn=predict_fake_news,
    inputs=gr.Textbox(lines=4, placeholder="Enter news content here...", label="📝 News Article Content"),
    outputs=gr.Textbox(label="📢 Prediction"),
    title=title,
    theme="default"
).launch()

# Save as app.py

from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route("/")
def home():
    return "✅ Fake Job Detection API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()


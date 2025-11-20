from flask import Flask, request, jsonify, render_template
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
    # Get JSON input
    data = request.get_json()
    description = data.get("text")

    # Transform using TF-IDF
    transformed_text = vectorizer.transform([description])

    # Predict
    prediction = model.predict(transformed_text)[0]
    probability = model.predict_proba(transformed_text)[0][1]

    # Prepare result
    result = {
        "prediction": "Fake Job" if prediction == 1 else "Real Job",
        "confidence": round(probability * 100, 2)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)

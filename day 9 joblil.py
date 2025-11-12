import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# ✅ Use joblib.load for both files
model = joblib.load("fake_job_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # safer: use .get() instead of ['description']
    description = request.form.get('description', '').strip()

    if not description:
        return render_template('index.html', error="Please enter a job description.")
    
    X = vectorizer.transform([description])
    prob = model.predict_proba(X)[0][1]
    label = "Fake Job" if prob > 0.5 else "Real Job"
    confidence = round(prob * 100, 2)

    return render_template('result.html', description=description, label=label, confidence=confidence)

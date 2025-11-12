from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer (with absolute paths)
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, "fake_job_model.pkl"))
vectorizer = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))

# Simple counters
fake_count = 0
real_count = 0

@app.route('/')
def home():
    return render_template('index.html', fake=fake_count, real=real_count)

@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count

    description = request.form.get('description', '').strip()

    # Validation
    if not description or len(description.split()) < 5:
        return render_template(
            'index.html',
            error="Please enter a detailed job description (at least 5 words).",
            fake=fake_count,
            real=real_count
        )

    # Predict
    X = vectorizer.transform([description])
    prob = model.predict_proba(X)[0][1]
    label = "Fake Job" if prob > 0.5 else "Real Job"
    confidence = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)

    # Update counters
    if label == "Fake Job":
        fake_count += 1
    else:
        real_count += 1

    return render_template(
        'result.html',
        description=description,
        label=label,
        confidence=confidence,
        fake=fake_count,
        real=real_count
    )

if __name__ == '__main__':
    app.run(debug=True)



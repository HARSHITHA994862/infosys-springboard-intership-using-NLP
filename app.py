from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import os
import sqlite3
import json   # ✅ REQUIRED for charts to work

app = Flask(__name__)
app.secret_key = "supersecretkey123"


# -----------------------------
# LOAD MODEL + VECTORIZER
# -----------------------------
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, "fake_job_model.pkl"))
vectorizer = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.pkl"))

fake_count = 0
real_count = 0


# -----------------------------
# INITIALIZE DATABASE
# -----------------------------
def init_db():
    conn = sqlite3.connect("job_predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            description TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()


# -----------------------------
# USER LOGIN
# -----------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get("username")
        password = request.form.get("password")

        # USER LOGIN (NOT ADMIN)
        if username == "user" and password == "user123":
            session['logged_in'] = True
            return redirect('/')
        else:
            return render_template("login.html", error="Invalid username or password!")

    return render_template("login.html")


# -----------------------------
# HOME PAGE
# -----------------------------
@app.route('/')
def home():
    if not session.get('logged_in'):
        return redirect('/login')

    return render_template(
        'index.html',
        fake=fake_count,
        real=real_count,
        last_prediction=None,
        error=None
    )


# -----------------------------
# PREDICTION
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    global fake_count, real_count

    if not session.get('logged_in'):
        return redirect('/login')

    description = request.form.get('job_description', '').strip()

    if not description or len(description.split()) < 5:
        return render_template(
            'index.html',
            fake=fake_count,
            real=real_count,
            last_prediction=None,
            error="Please enter at least 5 words."
        )

    # Predict
    X = vectorizer.transform([description])
    prob_fake = model.predict_proba(X)[0][1]

    confidence = prob_fake * 100 if prob_fake > 0.5 else (1 - prob_fake) * 100
    confidence = round(confidence, 2)

    THRESHOLD = 60
    is_fake = confidence < THRESHOLD or prob_fake > 0.5

    label = "Fake Job" if is_fake else "Real Job"

    if is_fake:
        fake_count += 1
    else:
        real_count += 1

    # Save to DB
    conn = sqlite3.connect("job_predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (description, prediction, confidence)
        VALUES (?, ?, ?)
    """, (description, label, confidence))
    conn.commit()
    conn.close()

    last_prediction = {"label": label, "confidence": confidence}

    return render_template(
        'index.html',
        fake=fake_count,
        real=real_count,
        last_prediction=last_prediction,
        error=None
    )


# -----------------------------
# LOGOUT
# -----------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')


# -----------------------------
# ADMIN LOGIN
# -----------------------------
@app.route('/admin_login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        if request.form.get("username") == "admin" and request.form.get("password") == "harshitha":
            session['admin_logged_in'] = True
            return redirect('/admin_dashboard')
        else:
            return render_template("admin_login.html", error="Incorrect Admin Credentials")

    return render_template("admin_login.html")


# -----------------------------
# ADMIN DASHBOARD (FIXED)
# -----------------------------
@app.route('/admin_dashboard')
def admin_dashboard():
    if not session.get('admin_logged_in'):
        return redirect('/admin_login')

    conn = sqlite3.connect("job_predictions.db")
    cursor = conn.cursor()

    fake_count = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction='Fake Job'"
    ).fetchone()[0]

    real_count = cursor.execute(
        "SELECT COUNT(*) FROM predictions WHERE prediction='Real Job'"
    ).fetchone()[0]

    daily_data = cursor.execute("""
        SELECT DATE(timestamp), COUNT(*)
        FROM predictions
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """).fetchall()

    dates = [row[0] for row in daily_data]
    counts = [row[1] for row in daily_data]

    conn.close()

    # 🔥 FIXED — Convert Python lists to JSON arrays for Chart.js
    return render_template(
        'admin_dashboard.html',
        fake=fake_count,
        real=real_count,
        dates=json.dumps(dates),    # <-- FIX
        counts=json.dumps(counts)   # <-- FIX
    )


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)









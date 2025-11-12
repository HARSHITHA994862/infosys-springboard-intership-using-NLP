# Day 8: Train – Evaluate – Save – Test Model

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import re

# -----------------------------------
# 1️⃣ Load Data
# -----------------------------------
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# ✅ Create clean text column if not present
if 'clean_description' not in df.columns:
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z ]', '', text)  # remove special chars & numbers
        text = re.sub(r'\s+', ' ', text)       # remove extra spaces
        return text
    
    df['clean_description'] = df['description'].apply(clean_text)

# Remove missing text rows
df = df.dropna(subset=['clean_description'])

# -----------------------------------
# 2️⃣ Vectorization (TF-IDF)
# -----------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_description'])
y = df['fraudulent']

# -----------------------------------
# 3️⃣ Train/Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# 4️⃣ Train Model
# -----------------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# -----------------------------------
# 5️⃣ Evaluate
# -----------------------------------
y_pred = model.predict(X_test)

print("\n📊 Model Performance")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------------
# 6️⃣ Save Model & Vectorizer
# -----------------------------------
joblib.dump(model, 'fake_job_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("\n✅ Model and Vectorizer Saved Successfully!")

# -----------------------------------
# 7️⃣ Prediction Demo
# -----------------------------------
print("\n🔍 Testing prediction on a sample text")

model = joblib.load('fake_job_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

sample_text = "Work from home job! Earn 50,000 per week. Limited seats. Apply fast."

x = vectorizer.transform([sample_text])
prob = model.predict_proba(x)[0][1]
label = "Fake Job" if prob > 0.5 else "Real Job"

print(f"\nText: {sample_text}")
print("Prediction:", label)
print("Probability:", round(prob, 4))


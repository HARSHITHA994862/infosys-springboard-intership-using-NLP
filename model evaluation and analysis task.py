# ==========================================================
# 🧩 Day 6: Baseline vs TF-IDF Model Comparison + Analysis
# ==========================================================

import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================================
# Step 1️⃣ – Load Dataset
# ==========================================================
file_path = r"C:\Users\Harsh\Downloads\fake_job_postings.csv"

if not os.path.exists(file_path):
    print(f"❌ File not found: {file_path}")
    exit()

df = pd.read_csv(file_path)

# ==========================================================
# Step 2️⃣ – Clean Data if Needed
# ==========================================================
if 'clean_description' not in df.columns:
    print("⚙️ Cleaning text data...")
    df = df.dropna(subset=['description'])

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_description'] = df['description'].apply(clean_text)
else:
    df = df.dropna(subset=['clean_description'])

X_text = df['clean_description']
y = df['fraudulent']

# ==========================================================
# Step 3️⃣ – Split Dataset
# ==========================================================
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# 🧩 Task 1 – Baseline Model (BoW) vs TF-IDF
# ==========================================================
print("\n==============================")
print("🧩 TASK 1: MODEL COMPARISON")
print("==============================")

# ---------- Bag of Words ----------
print("\n📦 Training Logistic Regression (CountVectorizer)...")
count_vectorizer = CountVectorizer(max_features=5000)
X_train_bow = count_vectorizer.fit_transform(X_train_text)
X_test_bow = count_vectorizer.transform(X_test_text)

model_bow = LogisticRegression(max_iter=200)
model_bow.fit(X_train_bow, y_train)
y_pred_bow = model_bow.predict(X_test_bow)

print("\n🔹 BoW Accuracy:", round(accuracy_score(y_test, y_pred_bow), 4))
print(classification_report(y_test, y_pred_bow))

# ---------- TF-IDF ----------
print("\n🔢 Training Logistic Regression (TF-IDF)...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

model_tfidf = LogisticRegression(max_iter=200)
model_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

print("\n🔹 TF-IDF Accuracy:", round(accuracy_score(y_test, y_pred_tfidf), 4))
print(classification_report(y_test, y_pred_tfidf))

# ---------- Comparison ----------
acc_bow = accuracy_score(y_test, y_pred_bow)
acc_tfidf = accuracy_score(y_test, y_pred_tfidf)

print("\n📈 Accuracy Comparison:")
print(f"   • BoW   : {acc_bow:.4f}")
print(f"   • TF-IDF: {acc_tfidf:.4f}")

if acc_tfidf > acc_bow:
    print("\n✅ TF-IDF performed better.")
    print("👉 Reason: TF-IDF reduces the weight of common words and emphasizes rare, discriminative terms.")
else:
    print("\n✅ BoW performed better (less common, but possible if dataset is small or noisy).")

# ==========================================================
# 🧩 Task 2 – Model Analysis (Fake Probability)
# ==========================================================
print("\n==============================")
print("🧩 TASK 2: MODEL ANALYSIS")
print("==============================")

# Use TF-IDF model for this analysis
df['predicted_proba'] = model_tfidf.predict_proba(tfidf_vectorizer.transform(df['clean_description']))[:, 1]

# Show top 5 most suspicious job postings
top_fake_jobs = df.sort_values(by='predicted_proba', ascending=False).head(5)
print("\n🔝 Top 5 jobs with highest fake probability:\n")
print(top_fake_jobs[['title', 'location', 'predicted_proba', 'clean_description']].to_string(index=False))

print("\n🕵️ Observations:")
print("→ These job posts likely contain suspicious keywords like 'urgent', 'limited vacancy', 'visa', 'investment', or 'fee'.")
print("→ Fake listings often promise unrealistic salaries or request money/training payments.")

# ==========================================================
# 🧩 Task 3 – TF-IDF Feature Size Sensitivity (Optional)
# ==========================================================
print("\n==============================")
print("🧩 TASK 3: TF-IDF Feature Comparison")
print("==============================")

for features in [1000, 5000, 10000]:
    tfidf = TfidfVectorizer(max_features=features)
    X_train_f = tfidf.fit_transform(X_train_text)
    X_test_f = tfidf.transform(X_test_text)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_f, y_train)
    preds = model.predict(X_test_f)
    acc = accuracy_score(y_test, preds)
    print(f"Features = {features:<6} → Accuracy = {acc:.4f}")

print("\n✅ Analysis complete!")

# ==========================================================
# 🧩 Optional Visualization – Confusion Matrix for TF-IDF
# ==========================================================
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_tfidf), annot=True, fmt='d',
            cmap='coolwarm', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix – Logistic Regression (TF-IDF)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

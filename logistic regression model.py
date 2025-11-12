# ==========================================================
# 🧩 Day 5: Logistic Regression Model for Fake Job Detection
# (Full version with auto data cleaning + error handling)
# ==========================================================

import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================================
# Step 1️⃣ – Load dataset safely
# ==========================================================
file_path = r"C:\Users\Harsh\Downloads\fake_job_postings.csv"   # 👈 update if stored elsewhere

if not os.path.exists(file_path):
    print(f"❌ File not found at: {file_path}")
    print("➡ Please check the path or move your dataset into this folder:")
    print(os.getcwd())
    exit()
else:
    print(f"✅ File found: {file_path}")

# Load dataset
df = pd.read_csv(file_path)

# ==========================================================
# Step 2️⃣ – Create clean_description if not found
# ==========================================================
if 'clean_description' not in df.columns:
    print("\n⚙️ Cleaning text data (since 'clean_description' not found)...")

    # Drop missing descriptions
    df = df.dropna(subset=['description'])

    # Function to clean job description text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)   # remove URLs
        text = re.sub(r'[^a-z\s]', '', text)                   # keep only letters/spaces
        text = re.sub(r'\s+', ' ', text).strip()               # remove extra spaces
        return text

    # Apply cleaning
    df['clean_description'] = df['description'].apply(clean_text)

    # Save cleaned version for later use
    cleaned_path = r"C:\Users\Harsh\Downloads\fake_job_postings_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"✅ Cleaned data saved at: {cleaned_path}")
else:
    print("✅ Using existing 'clean_description' column.")

# Drop rows with missing clean descriptions
df = df.dropna(subset=['clean_description'])

# ==========================================================
# Step 3️⃣ – Feature Extraction (TF-IDF)
# ==========================================================
print("\n📊 Extracting text features using TF-IDF...")
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_description'])
y = df['fraudulent']

# ==========================================================
# Step 4️⃣ – Split data into Train & Test sets
# ==========================================================
print("📂 Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================================
# Step 5️⃣ – Train Logistic Regression Model
# ==========================================================
print("🤖 Training Logistic Regression model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ==========================================================
# Step 6️⃣ – Make Predictions
# ==========================================================
y_pred = model.predict(X_test)

# ==========================================================
# Step 7️⃣ – Evaluate Model Performance
# ==========================================================
print("\n🔹 Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔹 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d',
            cmap='coolwarm', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ==========================================================
# Step 8️⃣ – Test Example Predictions
# ==========================================================
print("\n🧪 Testing model with example inputs...")

test_samples = [
    "Work from home! Limited vacancies. Apply now.",
    "We are hiring a data scientist for our Bangalore office."
]

sample_features = vectorizer.transform(test_samples)
sample_predictions = model.predict(sample_features)

print("\n🔹 Sample Predictions:")
for text, pred in zip(test_samples, sample_predictions):
    label = "FAKE" if pred == 1 else "REAL"
    print(f"→ '{text}' → {label}")

print("\n✅ Logistic Regression model executed successfully!")


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# ==============================
# Load dataset
# ==============================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Handle missing data
df['description'] = df['description'].fillna("")

# Define features and labels
X = df['description']
y = df['fraudulent']  # make sure this column exists (0=real, 1=fake)

# ==============================
# TF-IDF Vectorizer
# ==============================
vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_vec = vectorizer.fit_transform(X)

# ==============================
# Split and Train
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ==============================
# Save both model and vectorizer
# ==============================
joblib.dump(model, "fake_job_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("✅ Model and vectorizer retrained and saved successfully!")


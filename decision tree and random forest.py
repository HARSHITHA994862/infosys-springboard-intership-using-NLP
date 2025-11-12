# ===========================================
# 🧩 Day 6: Decision Tree & Random Forest Models
# ===========================================

# ====== Import Libraries ======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ===========================================
# 📘 Step 1: Load and Prepare the Dataset
# ===========================================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Drop rows with missing descriptions
df = df.dropna(subset=['description'])

# (Optional) Simple text cleaning – lowercase, remove special chars
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_description'] = df['description'].apply(clean_text)

# ===========================================
# 🔤 Step 2: TF-IDF Vectorization
# ===========================================
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_description'])
y = df['fraudulent']

# ===========================================
# ✂️ Step 3: Train-Test Split
# ===========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===========================================
# 🌳 Step 4: Decision Tree Model
# ===========================================
dt = DecisionTreeClassifier(max_depth=20, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

# ===========================================
# 🌲 Step 5: Random Forest Model
# ===========================================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# ===========================================
# 📊 Step 6: Model Evaluation
# ===========================================
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# ===========================================
# 🧮 Step 7: Confusion Matrix Visualization
# ===========================================
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ===========================================
# 🔎 Step 8: Feature Importance Visualization
# ===========================================
importances = rf.feature_importances_
indices = importances.argsort()[-10:][::-1]
feature_names = vectorizer.get_feature_names_out()

plt.figure(figsize=(8, 5))
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top 10 Important Words (Random Forest)")
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()
plt.show()


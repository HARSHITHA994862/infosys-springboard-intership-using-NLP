
# ====== Import Libraries ======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ===========================================
# 📘 Step 1: Load and Prepare the Dataset
# ===========================================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")
df = df.dropna(subset=['description'])

# Clean text if not already cleaned
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
# 🧠 Task 1: Model Comparison
# ===========================================

results = []

# --- Logistic Regression ---
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr)
})

# --- Decision Tree (varying max_depth) ---
for depth in [10, 20, 30]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    results.append({
        'Model': f'Decision Tree (max_depth={depth})',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

# --- Random Forest (varying n_estimators) ---
for n in [50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results.append({
        'Model': f'Random Forest (n={n})',
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    })

# Create DataFrame of results
results_df = pd.DataFrame(results)
print("\n=== Model Comparison Results ===")
print(results_df)

# ===========================================
# 📊 Plot Model Comparison
# ===========================================
plt.figure(figsize=(10, 5))
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')
plt.xticks(rotation=45, ha='right')
plt.title("Model Accuracy Comparison")
plt.tight_layout()
plt.show()
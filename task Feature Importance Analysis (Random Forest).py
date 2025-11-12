
# ===========================================
# 🧩 Day 6: Decision Tree & Random Forest Comparison + Feature Importance
# ===========================================

# ====== Import Libraries ======
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ===========================================
# 📘 Step 1: Load and Prepare the Dataset
# ===========================================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Drop missing descriptions
df = df.dropna(subset=['description'])

# Clean text
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
# 🧩 Task 1: Model Comparison
# ===========================================

results = []

# Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
results.append({
    'Model': 'Logistic Regression',
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-score': f1_score(y_test, y_pred_lr)
})

# Decision Tree (depth 10, 20, 30)
for depth in [10, 20, 30]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    results.append({
        'Model': f'Decision Tree (depth={depth})',
        'Accuracy': accuracy_score(y_test, y_pred_dt),
        'Precision': precision_score(y_test, y_pred_dt),
        'Recall': recall_score(y_test, y_pred_dt),
        'F1-score': f1_score(y_test, y_pred_dt)
    })

# Random Forest (estimators 50, 100, 200)
for n in [50, 100, 200]:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    results.append({
        'Model': f'Random Forest (n={n})',
        'Accuracy': accuracy_score(y_test, y_pred_rf),
        'Precision': precision_score(y_test, y_pred_rf),
        'Recall': recall_score(y_test, y_pred_rf),
        'F1-score': f1_score(y_test, y_pred_rf)
    })

# Convert to DataFrame for easy comparison
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df)

# 📊 Visualization
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1-score']),
            x='Model', y='value', hue='variable')
plt.xticks(rotation=45, ha='right')
plt.title("Model Comparison: Logistic Regression vs Decision Tree vs Random Forest")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend(title='Metric')
plt.tight_layout()
plt.show()

# ===========================================
# 🔎 Task 2: Feature Importance Analysis (Random Forest)
# ===========================================
best_rf = RandomForestClassifier(n_estimators=200, random_state=42)
best_rf.fit(X_train, y_train)

importances = best_rf.feature_importances_
indices = importances.argsort()[-15:][::-1]
feature_names = vectorizer.get_feature_names_out()

plt.figure(figsize=(8, 6))
plt.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.title("Top 15 Important Words (Random Forest)")
plt.xlabel("Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# Interpretation
print("\n=== Interpretation ===")
print("✅ The top 15 words above are most influential in predicting fake job postings.")
print("   Words like 'urgent', 'visa', 'investment', or 'training' often indicate suspicious jobs.")
print("⚠️ If unexpected words appear (e.g., 'manager', 'assistant'), they might co-occur in both real and fake ads.")

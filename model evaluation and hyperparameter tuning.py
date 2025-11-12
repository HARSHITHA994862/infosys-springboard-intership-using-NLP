# Day 7: Model Evaluation & Hyperparameter Tuning

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

# -------------------------------------------------------------
# 1. Load the cleaned dataset
# -------------------------------------------------------------
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Check if 'clean_description' exists, otherwise create one
if 'clean_description' not in df.columns:
    print("⚠️ 'clean_description' column not found. Creating one from 'description'...")
    df['clean_description'] = df['description'].fillna('').astype(str)

# Drop rows with missing cleaned descriptions
df = df.dropna(subset=['clean_description'])

# -------------------------------------------------------------
# 2. TF-IDF Vectorization
# -------------------------------------------------------------
print("🔹 Performing TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(df['clean_description'])
y = df['fraudulent']

# -------------------------------------------------------------
# 3. Train-Test Split
# -------------------------------------------------------------
print("🔹 Splitting dataset into Train/Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------
# 4. Initialize Models
# -------------------------------------------------------------
log_reg = LogisticRegression(max_iter=200, random_state=42)
rf = RandomForestClassifier(random_state=42)

# -------------------------------------------------------------
# 5. Cross-Validation (5-fold)
# -------------------------------------------------------------
print("🔹 Performing 5-Fold Cross-Validation...")
log_cv = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')
rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')

print(f"✅ Logistic Regression CV Accuracy: {log_cv.mean():.4f}")
print(f"✅ Random Forest CV Accuracy: {rf_cv.mean():.4f}")

# -------------------------------------------------------------
# 6. Fit Models on Training Data
# -------------------------------------------------------------
print("🔹 Training models...")
log_reg.fit(X_train, y_train)
rf.fit(X_train, y_train)

# -------------------------------------------------------------
# 7. ROC-AUC Comparison
# -------------------------------------------------------------
print("🔹 Generating ROC curves...")

y_prob_log = log_reg.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

fpr1, tpr1, _ = roc_curve(y_test, y_prob_log)
fpr2, tpr2, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 5))
plt.plot(fpr1, tpr1, label="Logistic Regression", linewidth=2)
plt.plot(fpr2, tpr2, label="Random Forest", linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()

print(f"🎯 Logistic Regression AUC: {roc_auc_score(y_test, y_prob_log):.4f}")
print(f"🎯 Random Forest AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

# -------------------------------------------------------------
# 8. Classification Report
# -------------------------------------------------------------
print("\n📊 Classification Report for Random Forest:")
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# -------------------------------------------------------------
# 9. Hyperparameter Tuning (GridSearchCV on Random Forest)
# -------------------------------------------------------------
print("🔹 Performing Hyperparameter Tuning (GridSearchCV)...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5],
}

grid = GridSearchCV(
    rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1
)

grid.fit(X_train, y_train)

print("\n🏆 Best Parameters Found:", grid.best_params_)
print(f"🏆 Best Cross-Validation Accuracy: {grid.best_score_:.4f}")

# -------------------------------------------------------------
# 10. Final Evaluation on Test Set
# -------------------------------------------------------------
best_rf = grid.best_estimator_
y_pred_best = best_rf.predict(X_test)
print("\n📊 Final Classification Report (Best Random Forest):")
print(classification_report(y_test, y_pred_best))

print("✅ Model Evaluation and Hyperparameter Tuning Completed Successfully!")

# -------------------------------------------------------------
# Day 7: Model Evaluation & Hyperparameter Tuning
# -------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import numpy as np

# -------------------------------------------------------------
# 1. Load Dataset
# -------------------------------------------------------------
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Check for cleaned column
if 'clean_description' not in df.columns:
    print("⚠️ 'clean_description' column not found. Creating one from 'description'...")
    df['clean_description'] = df['description'].fillna("").astype(str)

# -------------------------------------------------------------
# 2. TF-IDF Vectorization
# -------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
X = vectorizer.fit_transform(df['clean_description'])
y = df['fraudulent']

# -------------------------------------------------------------
# 3. Train-Test Split
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------------------
# Task 1: Cross-Validation Analysis
# -------------------------------------------------------------
print("\n🔹 Performing 5-Fold Cross-Validation...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

cv_means = {}
cv_stds = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_means[name] = scores.mean()
    cv_stds[name] = scores.std()
    print(f"{name}: Mean={scores.mean():.4f}, Std={scores.std():.4f}")

# Bar Chart for Mean CV Accuracy
plt.figure(figsize=(8,5))
plt.bar(cv_means.keys(), cv_means.values(), color=['skyblue','lightgreen','salmon'])
plt.title("Mean CV Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()

# Identify most stable model (lowest variance)
stable_model = min(cv_stds, key=cv_stds.get)
print(f"\n✅ Most stable model (lowest variance): {stable_model}")

# -------------------------------------------------------------
# Task 2: ROC-AUC Visualization
# -------------------------------------------------------------
print("\n🔹 Training models and plotting ROC curves...")

# Fit models
log_reg = LogisticRegression(max_iter=200).fit(X_train, y_train)
tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
rf = RandomForestClassifier(random_state=42).fit(X_train, y_train)

# Predict probabilities
y_prob_log = log_reg.predict_proba(X_test)[:, 1]
y_prob_tree = tree.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Compute ROC Curves
fpr1, tpr1, _ = roc_curve(y_test, y_prob_log)
fpr2, tpr2, _ = roc_curve(y_test, y_prob_tree)
fpr3, tpr3, _ = roc_curve(y_test, y_prob_rf)

# Plot ROC Curves
plt.figure(figsize=(8,6))
plt.plot(fpr1, tpr1, label=f"Logistic Regression (AUC={roc_auc_score(y_test, y_prob_log):.3f})")
plt.plot(fpr2, tpr2, label=f"Decision Tree (AUC={roc_auc_score(y_test, y_prob_tree):.3f})")
plt.plot(fpr3, tpr3, label=f"Random Forest (AUC={roc_auc_score(y_test, y_prob_rf):.3f})")
plt.plot([0,1],[0,1],'k--',label='Random Chance')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# Report AUCs
print("AUC Scores:")
print(f" - Logistic Regression: {roc_auc_score(y_test, y_prob_log):.3f}")
print(f" - Decision Tree:       {roc_auc_score(y_test, y_prob_tree):.3f}")
print(f" - Random Forest:       {roc_auc_score(y_test, y_prob_rf):.3f}")

# -------------------------------------------------------------
# Task 3: Hyperparameter Tuning (Decision Tree)
# -------------------------------------------------------------
print("\n🔹 Performing GridSearchCV for Decision Tree...")

param_grid = {
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                    param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)
print("Best Cross-Validation Accuracy:", grid.best_score_)

# Compare tuned Decision Tree vs Random Forest
tuned_tree = grid.best_estimator_
tuned_acc = tuned_tree.score(X_test, y_test)
rf_acc = rf.score(X_test, y_test)

print(f"\nTuned Decision Tree Accuracy: {tuned_acc:.4f}")
print(f"Random Forest Accuracy:       {rf_acc:.4f}")

print("\n✅ Observation:")
print("After tuning, the Decision Tree's accuracy improved due to optimal parameter selection "
      "that reduced overfitting and improved generalization. "
      "However, Random Forest still tends to perform better overall due to ensemble averaging.")

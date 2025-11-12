# ===========================================
# 🧩 Task 4: Keyword-Based Fake Job Detector
# ===========================================

import pandas as pd
import re

# ===============================
# Load the dataset
# ===============================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Ensure 'clean_description' column exists (replace if needed)
if 'clean_description' not in df.columns:
    df['clean_description'] = df['description'].fillna('').astype(str)

# ===============================
# Define suspicious keywords
# ===============================
suspicious_keywords = [
    "urgent",
    "work from home",
    "limited vacancy",
    "visa",
    "investment",
    "training fee",
    "money transfer",
    "registration fee",
    "send money",
    "pay to apply",
]

# ===============================
# Function: Rule-based flag
# ===============================
def rule_based_flag(text):
    """
    Returns 1 if any suspicious keyword appears in job description, else 0.
    """
    text = str(text).lower()
    for word in suspicious_keywords:
        if re.search(rf"\b{word}\b", text):
            return 1
    return 0

# Apply rule to all descriptions
df['suspect_flag'] = df['clean_description'].apply(rule_based_flag)

# ===============================
# Comparison with actual labels
# ===============================
print("\n===== Crosstab (Normalized) =====")
print(pd.crosstab(df['suspect_flag'], df['fraudulent'], normalize='all'))

# ===============================
# Display few suspect but not fraudulent
# ===============================
suspect_not_fake = df[(df['suspect_flag'] == 1) & (df['fraudulent'] == 0)]

print("\n===== 5 Examples of 'Suspect but Not Fake' =====")
for i, row in suspect_not_fake.head(5).iterrows():
    print(f"\nTitle: {row.get('title', 'N/A')}")
    print(f"Description: {row['clean_description'][:300]}...")
    print("-" * 80)

# ===============================
# Optional: Check logo presence pattern
# ===============================
print("\n===== Company Logo Distribution by Fraudulent Label =====")
print(df.groupby('fraudulent')['has_company_logo'].value_counts(normalize=True))

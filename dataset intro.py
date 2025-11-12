import pandas as pd

# ===============================
# Load the dataset
# ===============================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# ===============================
# Display sample data
# ===============================
print("===== SAMPLE DATA =====")
print(df.head())

# ===============================
# Dataset information
# ===============================
print("\n===== DATASET INFO =====")
df.info()

# ===============================
# Missing values per column
# ===============================
print("\n===== MISSING VALUES PER COLUMN =====")
print(df.isnull().sum())

# ===============================
# Target (fraudulent) distribution
# ===============================
print("\n===== TARGET (FRAUDULENT) DISTRIBUTION =====")
print(df['fraudulent'].value_counts())

# ===============================
# Dataset summary statistics
# ===============================
print("\n===== DATASET SUMMARY =====")
print(df.describe(include='all'))
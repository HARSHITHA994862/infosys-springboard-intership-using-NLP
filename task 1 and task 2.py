import pandas as pd

# ===============================
# Load the dataset
# ===============================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# ===============================
# Task 1: Basic dataset insights
# ===============================

# 1️⃣ Total number of records
print("===== TOTAL NUMBER OF RECORDS =====")
print(len(df))

# 2️⃣ Number of missing values per column
print("\n===== MISSING VALUES PER COLUMN =====")
print(df.isnull().sum())

# 3️⃣ Count of real vs fake jobs
print("\n===== COUNT OF REAL vs FAKE JOBS =====")
print(df['fraudulent'].value_counts())
print("\n0 = Real job, 1 = Fake job")

# 4️⃣ Display 3 examples of fake job descriptions
print("\n===== 3 EXAMPLES OF FAKE JOB DESCRIPTIONS =====")
fake_jobs = df[df['fraudulent'] == 1]
print(fake_jobs['description'].head(3))

# ===============================
# Task 2: Analytical questions
# ===============================

# 1️⃣ Feature with the most missing values
print("\n===== FEATURE WITH MOST MISSING VALUES =====")
most_missing = df.isnull().sum().idxmax()
print(f"The feature with the most missing values is: '{most_missing}'")

# 2️⃣ Explanation: Why job descriptions are more useful
print("\n===== WHY JOB DESCRIPTIONS ARE MORE USEFUL THAN TITLES =====")
explanation = """
Job descriptions provide detailed context — including responsibilities, requirements,
and language patterns — which can reveal suspicious or unrealistic claims.
Fake postings often include vague details, poor grammar, or exaggerated offers.
Job titles, on the other hand, are short and repetitive (e.g., 'Data Analyst', 'Sales Manager'),
so they carry less distinguishing information for detecting fraud.
"""
print(explanation)

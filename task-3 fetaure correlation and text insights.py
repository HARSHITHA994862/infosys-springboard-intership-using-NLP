# ===========================================
# 🧩 Task 3: Feature Correlation and Text Insights
# ===========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ===============================
# Load dataset
# ===============================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Ensure 'clean_description' column exists (replace if needed)
df['clean_description'] = df['description'].fillna('').astype(str)

# ===============================
# Select relevant columns
# ===============================
cols = ['has_company_logo', 'telecommuting', 'employment_type', 'required_experience', 'fraudulent']
df_subset = df[cols]

# ===============================
# Bar Chart 1: Company Logo vs Fraud
# ===============================
plt.figure(figsize=(6, 4))
sns.barplot(
    data=df_subset,
    x='has_company_logo',
    y='fraudulent',
    estimator=lambda x: sum(x == 1) / len(x),
    errorbar=None,
    palette='coolwarm'
)
plt.title("Proportion of Fake Jobs vs Company Logo Presence")
plt.xlabel("Has Company Logo (1 = Yes, 0 = No)")
plt.ylabel("Proportion of Fake Jobs")
plt.show()

# ===============================
# Bar Chart 2: Remote Work (Telecommuting)
# ===============================
plt.figure(figsize=(6, 4))
sns.barplot(
    data=df_subset,
    x='telecommuting',
    y='fraudulent',
    estimator=lambda x: sum(x == 1) / len(x),
    errorbar=None,
    palette='coolwarm'
)
plt.title("Proportion of Fake Jobs vs Remote Work Availability")
plt.xlabel("Telecommuting (1 = Remote, 0 = On-site)")
plt.ylabel("Proportion of Fake Jobs")
plt.show()

# ===============================
# Bar Chart 3: Employment Type
# ===============================
plt.figure(figsize=(8, 4))
sns.barplot(
    data=df_subset,
    x='employment_type',
    y='fraudulent',
    estimator=lambda x: sum(x == 1) / len(x),
    errorbar=None,
    palette='mako'
)
plt.title("Proportion of Fake Jobs vs Employment Type")
plt.xlabel("Employment Type")
plt.ylabel("Proportion of Fake Jobs")
plt.xticks(rotation=30)
plt.show()

# ===============================
# WordClouds for Text Insights
# ===============================
real_jobs = " ".join(df.loc[df['fraudulent'] == 0, 'clean_description'])
fake_jobs = " ".join(df.loc[df['fraudulent'] == 1, 'clean_description'])

# WordCloud for real job descriptions
plt.figure(figsize=(8, 5))
wordcloud_real = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(real_jobs)
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Real Job Descriptions")
plt.show()

# WordCloud for fake job descriptions
plt.figure(figsize=(8, 5))
wordcloud_fake = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(fake_jobs)
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud - Fake Job Descriptions")
plt.show()

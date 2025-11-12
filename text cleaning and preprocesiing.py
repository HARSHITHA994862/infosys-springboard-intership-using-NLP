# ================================================
# Day 3: Text Cleaning and Preprocessing
# ================================================
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===============================
# Download NLTK resources (run once)
# ===============================
nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# Load dataset (same as Day 2)
# ===============================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")  # Update path if needed

# ===============================
# Define text cleaning function
# ===============================
def clean_text(text):
    if pd.isnull(text):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Remove punctuation and numbers
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)

    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)

# ===============================
# Apply cleaning to key text column(s)
# ===============================
df['clean_description'] = df['description'].apply(clean_text)

# ===============================
# Show before and after cleaning
# ===============================
print("===== ORIGINAL TEXT (First 300 chars) =====")
print(df['description'].iloc[1][:300])

print("\n===== CLEANED TEXT (First 300 chars) =====")
print(df['clean_description'].iloc[1][:300])

# ===============================
# Example of cleaned data (first 3 rows)
# ===============================
print("\n===== EXAMPLE OF CLEANED DATA =====")
print(df[['description', 'clean_description']].head(3))

# ===============================
# Save cleaned dataset for Day 4
# ===============================
output_path = r"C:\Users\Harsh\Downloads\clean_fake_job_postings.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved successfully at: {output_path}")


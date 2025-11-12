import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ================================
# Load your dataset
# ================================
# (Change the path if your file is elsewhere)
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# ================================
# Text Cleaning Function
# ================================
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)  # remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
        return text
    return ""

# Apply cleaning to description column
df['clean_description'] = df['description'].apply(clean_text)

# ================================
# Initialize and Fit TF-IDF Vectorizer
# ================================
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,   # you can adjust this number
    ngram_range=(1,2)    # include unigrams and bigrams
)

X = vectorizer.fit_transform(df['clean_description'])

# ================================
# Save the TF-IDF Vectorizer
# ================================
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

print("✅ TF-IDF vectorizer created and saved successfully as 'tfidf_vectorizer.pkl'!")

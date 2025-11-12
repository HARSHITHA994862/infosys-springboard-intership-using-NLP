import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ----------------------------
# Download NLTK data
# ----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# ----------------------------
# Initialize stopwords and lemmatizer
# ----------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ----------------------------
# Text cleaning function
# ----------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)               # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)      # Remove URLs
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation and digits
    text = re.sub(r'\s+', ' ', text).strip()        # Remove extra spaces
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# ----------------------------
# Task 1: BoW and TF-IDF for company_profile
# ----------------------------
df['clean_company'] = df['company_profile'].apply(clean_text)
company_texts = df['clean_company'].fillna('').tolist()

# Bag-of-Words
bow_vectorizer_company = CountVectorizer(max_features=2000)
X_bow_company = bow_vectorizer_company.fit_transform(company_texts)
print("BoW shape (company_profile):", X_bow_company.shape)

# TF-IDF
tfidf_vectorizer_company = TfidfVectorizer(max_features=2000)
X_tfidf_company = tfidf_vectorizer_company.fit_transform(company_texts)
print("TF-IDF shape (company_profile):", X_tfidf_company.shape)

print("\nNote: TF-IDF usually captures meaning better because frequent but uninformative words are down-weighted.")

# ----------------------------
# Task 2: Top 20 most frequent words in job descriptions
# ----------------------------
df['clean_description'] = df['description'].apply(clean_text)
description_texts = df['clean_description'].fillna('').tolist()

bow_vectorizer_desc = CountVectorizer()
X_bow_desc = bow_vectorizer_desc.fit_transform(description_texts)

# Count word occurrences
word_counts = X_bow_desc.sum(axis=0)
words = bow_vectorizer_desc.get_feature_names_out()
word_freq = [(word, word_counts[0, idx]) for idx, word in enumerate(words)]

# Sort by frequency
word_freq_sorted = sorted(word_freq, key=lambda x: x[1], reverse=True)

# Print top 20 words
print("\nTop 20 most frequent words in job descriptions:")
for word, freq in word_freq_sorted[:20]:
    print(f"{word}: {freq}")

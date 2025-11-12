import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")

# Initialize stopwords and lemmatizer once
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Define text cleaning function
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)              # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)     # Remove URLs
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)  # Remove punctuation and digits
    text = re.sub(r'\s+', ' ', text).strip()       # Remove extra spaces
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# Apply cleaning to the 'description' column
df['clean_description'] = df['description'].apply(clean_text)

# Convert cleaned descriptions to list
texts = df['clean_description'].fillna('').tolist()

# -------------------------------
# 1️⃣ Bag-of-Words (BoW)
# -------------------------------
bow_vectorizer = CountVectorizer(max_features=2000)  # Limit to top 2000 words
X_bow = bow_vectorizer.fit_transform(texts)

print("BoW shape:", X_bow.shape)
print("Sample feature names (BoW):", bow_vectorizer.get_feature_names_out()[:10])
print("Example BoW vector (first row):")
print(X_bow[0].toarray())

# -------------------------------
# 2️⃣ TF-IDF
# -------------------------------
tfidf_vectorizer = TfidfVectorizer(max_features=2000)
X_tfidf = tfidf_vectorizer.fit_transform(texts)

print("\nTF-IDF shape:", X_tfidf.shape)
print("Sample feature names (TF-IDF):", tfidf_vectorizer.get_feature_names_out()[:10])
print("Example TF-IDF vector (first row):")
print(X_tfidf[0].toarray())
 
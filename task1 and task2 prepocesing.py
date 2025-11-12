import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# ===============================
# Load your dataset here
# ===============================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")   

# ===============================
# Define cleaning function
# ===============================
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

# ===============================
# Apply cleaning
# ===============================
df['clean_description'] = df['description'].apply(clean_text)

print(df[['description', 'clean_description']].head())



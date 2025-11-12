import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary resources
nltk.download('stopwords')
nltk.download('wordnet')

# ==========================
# Load the dataset
# ==========================
df = pd.read_csv(r"C:\Users\Harsh\Downloads\fake_job_postings.csv")  # 🔁 change filename if needed

# ==========================
# Define cleaning function
# ==========================
def clean_company_profile(text):
    if pd.isnull(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove numbers and punctuation
    text = re.sub(r'[%s\d]' % re.escape(string.punctuation), ' ', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)

# ==========================
# Apply cleaning
# ==========================
df['clean_company_profile'] = df['company_profile'].apply(clean_company_profile)

# Display results
print(df[['company_profile', 'clean_company_profile']].head())
# Function to count words in a text
def word_count(text):
    return len(str(text).split())

# Apply to job descriptions
df['desc_word_count_before'] = df['description'].apply(word_count)
df['desc_word_count_after'] = df['clean_company_profile'].apply(word_count)

# Calculate averages
avg_before = df['desc_word_count_before'].mean()
avg_after = df['desc_word_count_after'].mean()

print(f"Average word count before cleaning: {avg_before:.2f}")
print(f"Average word count after cleaning:  {avg_after:.2f}")
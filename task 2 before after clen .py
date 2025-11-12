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

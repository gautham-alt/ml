from sklearn.feature_extraction.text import CountVectorizer

# Sample text documents
text = [
    "Machine learning is interesting",
    "Machine learning is useful",
    "Python is interesting"
]

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Convert text into numerical vectors
X = vectorizer.fit_transform(text)

# Display vocabulary
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# Display vectorized data
print("Vectorized Data:")
print(X.toarray())

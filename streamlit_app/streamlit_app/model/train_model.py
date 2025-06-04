import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import pickle
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string

# Load data
fake = pd.read_csv("model/Fake.csv")
true = pd.read_csv("model/True.csv")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
data = pd.concat([fake, true], axis=0)
data = data[["text", "label"]]

# Clean text
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

data["text"] = data["text"].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier()
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model

with open("model/model.pkl", "wb") as f:


    pickle.dump(model, f)

with open("model/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

import pandas as pd
import re
import pickle
import nltk
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(__file__)

# Load datasets
true_df = pd.read_csv(os.path.join(BASE_DIR, "true.csv"))
fake_df = pd.read_csv(os.path.join(BASE_DIR, "fake.csv"))

# Auto-detect text column
TEXT_COL = true_df.columns[0]

true_df = true_df[[TEXT_COL]]
fake_df = fake_df[[TEXT_COL]]

true_df['label'] = 1
fake_df['label'] = 0

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
df.rename(columns={TEXT_COL: 'text'}, inplace=True)

# Balance dataset
real_df = df[df['label'] == 1]
fake_df = df[df['label'] == 0]

min_size = min(len(real_df), len(fake_df))

real_df = real_df.sample(min_size, random_state=42)
fake_df = fake_df.sample(min_size, random_state=42)

df = pd.concat([real_df, fake_df]).sample(frac=1, random_state=42)

# Improved text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

# TF-IDF (Improved)
vectorizer = TfidfVectorizer(
    max_features=2500,      # was 15000
    ngram_range=(1,1),      # remove bigrams
    min_df=2
)
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Stronger Logistic Regression
model = LogisticRegression(
    max_iter=500,
    solver="liblinear"
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Improved Model Accuracy: {accuracy * 100:.2f}%")

# Save model
pickle.dump(model, open(os.path.join(BASE_DIR, "model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb"))

print("Model and vectorizer saved successfully.")

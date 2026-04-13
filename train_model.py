import pandas as pd
import re
import pickle
import os
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

BASE_DIR = os.path.dirname(__file__)

# -----------------------------
# LOAD DATASETS
# -----------------------------
true_df = pd.read_csv(os.path.join(BASE_DIR, "True.csv"))
fake_df = pd.read_csv(os.path.join(BASE_DIR, "Fake.csv"))

# Auto-detect first text column
TEXT_COL = true_df.columns[0]

true_df = true_df[[TEXT_COL]]
fake_df = fake_df[[TEXT_COL]]

true_df["label"] = 1
fake_df["label"] = 0

# -----------------------------
# COMBINE + BALANCE
# -----------------------------
df = pd.concat([true_df, fake_df], ignore_index=True)
df.rename(columns={TEXT_COL: "text"}, inplace=True)

real_df = df[df["label"] == 1]
fake_only_df = df[df["label"] == 0]

min_size = min(len(real_df), len(fake_only_df))

real_df = real_df.sample(min_size, random_state=42)
fake_only_df = fake_only_df.sample(min_size, random_state=42)

df = pd.concat([real_df, fake_only_df]).sample(frac=1, random_state=42)

# -----------------------------
# ADVANCED CLEANING
# -----------------------------
def clean_text(text):
    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\\S+|www\\S+", "", text)

    # keep only alphabets
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)

    # tokenize
    words = text.split()

    # remove stopwords + stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(words)

df["text"] = df["text"].apply(clean_text)

# -----------------------------
# BETTER TF-IDF
# -----------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# STRONGER MODEL (BEST FOR NLP)
# -----------------------------
model = LinearSVC(C=1.5)

model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"🚀 Optimized Model Accuracy: {accuracy * 100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
with open(os.path.join(BASE_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(BASE_DIR, "vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print("\n✅ Optimized model and vectorizer saved successfully.")

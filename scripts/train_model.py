import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Učitavanje podataka
df = pd.read_csv("data/products.csv")

# Čišćenje imena kolona
df.columns = df.columns.str.strip()

# Uklanjanje redova sa praznim vrednostima u ključnim kolonama
df = df.dropna(subset=["Product Title", "Category Label"])

# Definisanje X i y
X = df["Product Title"]
y = df["Category Label"]

# Kreiranje pipeline-a
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", RandomForestClassifier())
])

# Treniranje modela
pipeline.fit(X, y)

# Čuvanje modela
joblib.dump(pipeline, "models/product_classifier.pkl", compress=3)
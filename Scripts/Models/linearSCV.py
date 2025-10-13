import pandas as pd
import numpy as np
import joblib
import argparse
import os
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

RSEED = 42

# ==========================
# 0. Argument Parsing
# ==========================
parser = argparse.ArgumentParser(description="Train a LinearSVC text classifier for chef prediction")
parser.add_argument("train_csv", type=str, help="Path to training CSV file")
parser.add_argument("test_csv", type=str, help="Path to testing CSV file")
args = parser.parse_args()

train_path = args.train_csv
test_path = args.test_csv

# Create results filename
test_base = os.path.splitext(os.path.basename(test_path))[0]
results_name = f"../Results/results_linearSVC_{test_base}.txt"

# ==========================
# 1. Load and preprocess data
# ==========================
train = pd.read_csv(train_path, sep=";")
test = pd.read_csv(test_path, sep=";")

def combine_fields(row):
    return " ".join([
        str(row.get(k, "")) 
        for k in ["recipe_name", "tags", "steps", "description", "ingredients", "data", "n_ingredients"]
    ])

train["text"] = train.apply(combine_fields, axis=1).astype(str)
X = train["text"]
y = train["chef_id"].astype(str)

# ==========================
# 2. TF-IDF + LinearSVC Pipeline
# ==========================
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode")),
    ("clf", LinearSVC(random_state=RSEED))
])

# Grid search for the best TF-IDF + SVC parameters
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2, 5],
    "tfidf__max_df": [0.9, 1.0],
    "clf__C": [0.5, 1.0, 2.0]
}

print("🔍 Running GridSearchCV for LinearSVC...")
gs = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=5, n_jobs=-1, verbose=1)
gs.fit(X, y)

print(f"✅ Best params: {gs.best_params_}")
print(f"✅ CV macro-F1: {gs.best_score_:.4f}")

best_model = gs.best_estimator_

# ==========================
# 3. Train on full data + predict
# ==========================
print("🚀 Training final model on full dataset...")
best_model.fit(X, y)
joblib.dump(best_model, "chef_model.pkl")

test["text"] = test.apply(combine_fields, axis=1).astype(str)
preds = best_model.predict(test["text"])

# ==========================
# 4. Save predictions
# ==========================
with open(results_name, "w", encoding="utf-8", newline="\n") as f:
    f.write("chef_id\n")
    for x in preds:
        f.write(f"{x}\n")

print(f"✅ Results saved to {results_name}")


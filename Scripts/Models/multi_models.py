import pandas as pd, numpy as np, joblib
import argparse
import os
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

RSEED = 42

# ==========================
# 0. Argument Parsing
# ==========================
parser = argparse.ArgumentParser(description="Train an RNN classifier on recipes")
parser.add_argument("train_csv", type=str, help="Path to training CSV file")
parser.add_argument("test_csv", type=str, help="Path to testing CSV file")
args = parser.parse_args()

train_path = args.train_csv
test_path = args.test_csv

# Create results filename
test_base = os.path.splitext(os.path.basename(test_path))[0]
results_name = f"./Results/results_linearSVC_{test_base}.txt"


train = pd.read_csv(train_path, sep=";")
test = pd.read_csv(test_path, sep=";")


def combine_fields(row):
    return " ".join([str(row.get(k,"")) for k in ["recipe_name","tags","steps","description","ingredients", "data", "n_ingredients"]])

train["text"] = train.apply(combine_fields, axis=1).astype(str)
X = train["text"]
y = train["chef_id"].astype(str)

# 1) Multi-model 5×2 CV comparison
tfidf = TfidfVectorizer(
    max_features=50000, ngram_range=(1,2),
    min_df=2, max_df=1.0,
    sublinear_tf=True, strip_accents="unicode"
)
cands = {
    "LinearSVC": LinearSVC(C=1.0, random_state=RSEED),
    "LogReg": LogisticRegression(max_iter=1000, solver="saga", C=1.0, n_jobs=-1, random_state=RSEED),
    "ComplNB": ComplementNB()
}
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RSEED)
rows = []
for name, clf in cands.items():
    pipe = Pipeline([("tfidf", tfidf), ("clf", clf)])
    cv = cross_validate(pipe, X, y, cv=rkf,
                        scoring={"acc":"accuracy","f1":"f1_macro"},
                        n_jobs=-1)
    rows.append((name, cv["test_acc"].mean(), cv["test_acc"].std(),
                 cv["test_f1"].mean(), cv["test_f1"].std()))
print("Model  acc_mean  acc_std  f1_macro_mean  f1_macro_std")
for r in rows:
    print(f"{r[0]:<9} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}")

# 2) Small grid search for LinearSVC
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(sublinear_tf=True, strip_accents="unicode")),
    ("clf", LinearSVC(random_state=RSEED))
])
param_grid = {
    "tfidf__ngram_range": [(1,1),(1,2)],
    "tfidf__min_df": [1,2,5],
    "tfidf__max_df": [0.9,1.0],
    "clf__C": [0.5,1.0,2.0]
}
gs = GridSearchCV(pipe, param_grid, scoring="f1_macro", cv=5, n_jobs=-1, verbose=1)
gs.fit(X, y)
print("Best params:", gs.best_params_)
print("CV macro-F1:", round(gs.best_score_,4))
best_model = gs.best_estimator_

# 3) Full training + generating results
best_model.fit(X, y)
joblib.dump(best_model, "chef_model.pkl")
test["text"] = test.apply(combine_fields, axis=1).astype(str)
preds = best_model.predict(test["text"])

with open(results_name, "w", encoding="utf-8", newline="\n") as f:
    f.write("chef_id\n")
    for x in preds:
        f.write(f"{x}\n")

print(f"✅ {results_name} has been generated")



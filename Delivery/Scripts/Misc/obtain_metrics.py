import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Parse command-line arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Compute accuracy, F1-score, and confusion matrix of predicted chef_ids")
parser.add_argument("results_txt", type=str, help="Path to the results.txt file (predicted chef_ids)")
parser.add_argument("testing_csv", type=str, help="Path to the testing CSV file (with actual chef_ids)")
args = parser.parse_args()

results_path = args.results_txt
testing_path = args.testing_csv

# ----------------------------
# Load predicted chef_ids
# ----------------------------
with open(results_path, "r") as f:
    lines = f.readlines()

# Skip header if present
if lines[0].strip().lower() == "chef_id":
    lines = lines[1:]

predicted = [line.strip() for line in lines]

# ----------------------------
# Load actual chef_ids
# ----------------------------
df_test = pd.read_csv(testing_path, sep=";")
if "chef_id" not in df_test.columns:
    raise ValueError("The testing CSV does not contain a 'chef_id' column.")
actual = df_test["chef_id"].astype(str).tolist()

# ----------------------------
# Check lengths match
# ----------------------------
if len(predicted) != len(actual):
    raise ValueError(f"Length mismatch: {len(predicted)} predictions vs {len(actual)} actual labels.")

# ----------------------------
# Compute metrics
# ----------------------------
acc = accuracy_score(actual, predicted)
f1_macro = f1_score(actual, predicted, average="macro")
f1_weighted = f1_score(actual, predicted, average="weighted")

print(f"🎯 Accuracy: {acc:.4f}")
print(f"📊 F1-score (macro): {f1_macro:.4f}")
print(f"📊 F1-score (weighted): {f1_weighted:.4f}")

# ----------------------------
# Confusion matrix
# ----------------------------
labels = sorted(set(actual))
cm = confusion_matrix(actual, predicted, labels=labels)

print("\n🧩 Confusion Matrix (rows=actual, cols=predicted):")
print(pd.DataFrame(cm, index=labels, columns=labels))

# ----------------------------
# Plot normalized confusion matrix
# ----------------------------
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Confusion Matrix - LinearSVC (Normalized)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.tight_layout()
plt.savefig("confusion_matrix_svc.png", dpi=300)
plt.show()

# ----------------------------
# Detailed per-class report
# ----------------------------
print("\n📋 Classification Report:")
print(classification_report(actual, predicted, digits=4))

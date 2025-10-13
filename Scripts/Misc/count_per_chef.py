import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ==========================
# Load data and encode chefs
# ==========================
train_df = pd.read_csv("../Datasets/OGs/train.csv", sep=";")
label_enc = LabelEncoder()
y = label_enc.fit_transform(train_df["chef_id"])

length = len(y)
amounts = [0] * len(label_enc.classes_)
total = len(y)

for i in range(length):
    amounts[y[i]] += 1

percentages = [round(count / total * 100, 2) for count in amounts]

# ==========================
# Print counts and percentages
# ==========================
chefs = list(label_enc.classes_)
chefs = [ str(x) for x in chefs ]
print("Chef IDs:", chefs)
print("Counts:", amounts)
print("Percentages:", percentages)

# ==========================
# Plot percentages
# ==========================

fig, ax = plt.subplots()

ax.bar(chefs, percentages)

ax.set_ylabel('percentage (%)')
ax.set_title('% of Recipes per Chef_ID')

plt.show()

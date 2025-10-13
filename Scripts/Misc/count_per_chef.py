import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_df=pd.read_csv("../train.csv", sep=";")
label_enc = LabelEncoder()
y = label_enc.fit_transform(train_df["chef_id"])

length = len(y)
amounts = [0,0,0,0,0,0]

for i in range(length):
    amounts[y[i]] += 1

print(amounts)

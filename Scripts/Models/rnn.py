import pandas as pd
import numpy as np
import torch
import argparse
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer
from collections import Counter


# ==========================
# 0. Argument Parsing
# ==========================
parser = argparse.ArgumentParser(description="Train an RNN classifier on recipes")

# Existing arguments
parser.add_argument("train_csv", type=str, help="Path to training CSV file")
parser.add_argument("test_csv", type=str, help="Path to testing CSV file")

# ✅ New arguments
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size of the RNN")
parser.add_argument("--max_len", type=int, default=100, help="Maximum sequence length for padding/truncation")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")

args = parser.parse_args()

train_path = args.train_csv
test_path = args.test_csv
HIDDEN_DIM = args.hidden_dim
MAX_LEN = args.max_len
EPOCHS = args.epochs

# Create results filename
test_base = os.path.splitext(os.path.basename(test_path))[0]
results_name = f"../Results/results_rnn_{test_base}.txt"

# ==========================
# 1. Data Loading & Preprocessing
# ==========================
def combine_fields(row):
    return " ".join([
        str(row["recipe_name"]),
        str(row["tags"]),
        str(row["steps"]),
        str(row["description"]),
        str(row["ingredients"])
    ])

train_df = pd.read_csv(train_path, sep=";")
test_df = pd.read_csv(test_path, sep=";")

train_df["text"] = train_df.apply(combine_fields, axis=1)
test_df["text"] = test_df.apply(combine_fields, axis=1)

# Encode chef labels
label_enc = LabelEncoder()
y = label_enc.fit_transform(train_df["chef_id"])

# Split train/val
X_train, X_val, y_train, y_val = train_test_split(
    train_df["text"].values, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================
# 2. Tokenization & Vocab
# ==========================
tokenizer = get_tokenizer("basic_english")

counter = Counter()
for text in X_train:
    counter.update(tokenizer(text))

vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(20000))}
vocab["<unk>"] = 1
vocab["<pad>"] = 0

VOCAB_SIZE = len(vocab)
NUM_CLASSES = len(label_enc.classes_)

def encode_text(text):
    tokens = tokenizer(text)
    ids = [vocab.get(tok, 1) for tok in tokens[:MAX_LEN]]
    if len(ids) < MAX_LEN:
        ids += [0] * (MAX_LEN - len(ids))
    return ids

X_train_enc = [encode_text(t) for t in X_train]
X_val_enc = [encode_text(t) for t in X_val]
X_test_enc = [encode_text(t) for t in test_df["text"]]

# ==========================
# 3. Dataset & DataLoader
# ==========================
class RecipeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

train_ds = RecipeDataset(X_train_enc, y_train)
val_ds = RecipeDataset(X_val_enc, y_val)
test_ds = RecipeDataset(X_test_enc)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# ==========================
# 4. RNN Model
# ==========================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2,:,:], h[-1,:,:]), dim=1)  # bidirectional concat
        h = self.dropout(h)
        return self.fc(h)

# ==========================
# 5. Training Loop
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNClassifier(VOCAB_SIZE, 128, HIDDEN_DIM, NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for Xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(Xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}")

    # Evaluate
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            targets.extend(yb.numpy())
    acc = accuracy_score(targets, preds)
    print(f"Validation Accuracy: {acc:.4f}")

# ==========================
# 6. Predict on Test & Save
# ==========================
model.eval()
pred_labels = []
with torch.no_grad():
    for Xb in test_loader:
        Xb = Xb.to(device)
        out = model(Xb)
        pred_labels.extend(torch.argmax(out, 1).cpu().numpy())

chef_ids = label_enc.inverse_transform(pred_labels)

with open(results_name, "w") as f:
    f.write("chef_id\n")  # write header
    np.savetxt(f, chef_ids, fmt="%s")

print(f"Saved predictions to {results_name} ✅")

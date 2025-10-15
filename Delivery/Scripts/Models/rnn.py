# improved_rnn_runner.py
import random
import os
import json
import math
import time
from collections import Counter
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ==========================
# 0. Argument Parsing
# ==========================
parser = argparse.ArgumentParser(description="Train an RNN classifier on recipes")
parser.add_argument("train_csv", type=str, help="Path to training CSV file")
parser.add_argument("test_csv", type=str, help="Path to testing CSV file")
parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size of the RNN")
parser.add_argument("--max_len", type=int, default=100, help="Maximum sequence length for padding/truncation")
parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
args = parser.parse_args()

# -------------------------
# Config / Hyperparameters
# -------------------------
SEED = 42
TRAIN_CSV = args.train_csv
TEST_CSV = args.test_csv
TOP_K = 30000
MAX_LEN = args.max_len
EMBED_DIM = 200            # fixed typo (was 200256)
HIDDEN_DIM = args.hidden_dim
BATCH_SIZE = 64
EPOCHS = args.epochs
LR = 3e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 3
CLIP_GRAD = 1.0
USE_AMP = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)

# -------------------------
# 1) Data loading & basic preprocessing
# -------------------------
def combine_fields(row):
    return " ".join([
        str(row.get('recipe_name', '')),
        str(row.get('tags', '')),
        str(row.get('steps', '')),
        str(row.get('description', '')),
        str(row.get('ingredients', ''))
    ])

print('Loading data...')
train_df = pd.read_csv(TRAIN_CSV, sep=';')
test_df = pd.read_csv(TEST_CSV, sep=';')

train_df['text'] = train_df.apply(combine_fields, axis=1)
test_df['text'] = test_df.apply(combine_fields, axis=1)

label_enc = LabelEncoder()
y = label_enc.fit_transform(train_df['chef_id'])

X_train_text, X_val_text, y_train, y_val = train_test_split(
    train_df['text'].values, y, test_size=0.2, random_state=SEED, stratify=y
)

# -------------------------
# 2) Tokenization & Vocab
# -------------------------
from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')

counter = Counter()
for t in X_train_text:
    counter.update(tokenizer(str(t)))

most_common = [w for w, _ in counter.most_common(TOP_K)]
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for i, w in enumerate(most_common, start=2):
    vocab[w] = i

VOCAB_SIZE = len(vocab)
NUM_CLASSES = len(label_enc.classes_)
print(f'Vocab size: {VOCAB_SIZE}, #classes: {NUM_CLASSES}')

def encode_tokens(text):
    return [vocab.get(t, vocab[UNK_TOKEN]) for t in tokenizer(str(text))]

X_train_tok = [encode_tokens(t) for t in X_train_text]
X_val_tok = [encode_tokens(t) for t in X_val_text]
X_test_tok = [encode_tokens(t) for t in test_df['text'].values]

# -------------------------
# 3) Dataset & collate_fn
# -------------------------
class RecipeDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        seq = torch.tensor(self.X[idx], dtype=torch.long)
        if self.y is not None:
            return seq, torch.tensor(int(self.y[idx]), dtype=torch.long)
        return seq

def collate_batch(batch):
    """
    Batch can be either list of tensors (test) or list of (tensor, label).
    Returns:
        padded: LongTensor (batch, seq_len)
        lengths: LongTensor (batch,)
        labels: LongTensor (batch,) or None
    """
    has_labels = isinstance(batch[0], tuple)
    if has_labels:
        sequences, labels = zip(*batch)
        labels = torch.stack(labels)
    else:
        sequences = batch
        labels = None

    # truncate sequences to MAX_LEN and compute lengths
    seqs = [s[:MAX_LEN] for s in sequences]
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)

    # pad to longest in batch (<= MAX_LEN)
    padded = pad_sequence(seqs, batch_first=True, padding_value=vocab[PAD_TOKEN])
    if padded.size(1) > MAX_LEN:
        padded = padded[:, :MAX_LEN]
        lengths = torch.clamp(lengths, max=MAX_LEN)

    # sort by length desc for pack_padded_sequence
    lengths, perm_idx = lengths.sort(0, descending=True)
    padded = padded[perm_idx]
    if labels is not None:
        labels = labels[perm_idx]

    if labels is not None:
        return padded, lengths, labels
    else:
        return padded, lengths

# DataLoaders (use collate_fn)
train_dataset = RecipeDataset(X_train_tok, y_train)
val_dataset = RecipeDataset(X_val_tok, y_val)
test_dataset = RecipeDataset(X_test_tok)

dl_kwargs = dict(batch_size=BATCH_SIZE, collate_fn=collate_batch, num_workers=0, pin_memory=(DEVICE.type=='cuda'))
train_loader = DataLoader(train_dataset, shuffle=True, **dl_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **dl_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **dl_kwargs)

# -------------------------
# 4) Improved RNN with attention
# -------------------------
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)
    def forward(self, outputs, lengths):
        # outputs: (batch, seq_len, hidden*2)
        weights = self.attn(outputs).squeeze(-1)  # (batch, seq_len)
        max_len = outputs.size(1)
        device = outputs.device
        # broadcasting approach for mask
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        weights = weights.masked_fill(~mask, float('-inf'))
        attn_scores = torch.softmax(weights, dim=1).unsqueeze(-1)
        pooled = (outputs * attn_scores).sum(dim=1)
        return pooled

class ImprovedRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=1, embedding_dropout=0.2, rnn_dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab[PAD_TOKEN])
        self.embed_drop = nn.Dropout(embedding_dropout)
        # only apply dropout in RNN if num_layers > 1 (to avoid warning)
        rnn_dropout = rnn_dropout if num_layers > 1 else 0.0
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=rnn_dropout, num_layers=num_layers)
        self.attn = AttentionPooling(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x, lengths):
        emb = self.embedding(x)
        emb = self.embed_drop(emb)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        pooled = self.attn(out, lengths)
        return self.fc(pooled)

# -------------------------
# 5) Loss, optimizer, scheduler, class weights
# -------------------------
class_counts = Counter(y_train)
counts = np.array([class_counts[i] for i in range(NUM_CLASSES)], dtype=np.float32)
class_weights = torch.tensor(counts.max() / counts, dtype=torch.float32).to(DEVICE)

model = ImprovedRNN(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES, num_layers=1).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE.type=='cuda'))

# -------------------------
# 6) Training loop
# -------------------------
def train_one_epoch(loader, model, criterion, optimizer, scaler, device=DEVICE, use_amp=True, clip_grad=1.0, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc='Train', leave=False), start=1):
        # batch comes either as (padded, lengths, labels) or (padded, lengths)
        if len(batch) == 3:
            Xb, lengths, yb = batch
        else:
            Xb, lengths = batch
            yb = None

        Xb = Xb.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True) if yb is not None else None

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(use_amp and device.type=='cuda')):
            out = model(Xb, lengths)
            loss = criterion(out, yb) if yb is not None else torch.tensor(0.0, device=device)

        # scale and backward
        if yb is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * Xb.size(0)

    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    torch.cuda.empty_cache()
    return avg_loss

def evaluate(loader, model, device=DEVICE):
    model.eval()
    preds = []
    targets = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                Xb, lengths, yb = batch
            else:
                Xb, lengths = batch
                yb = None

            Xb = Xb.to(device)
            lengths = lengths.to(device)
            if yb is not None:
                yb = yb.to(device)   # ✅ move labels to GPU here!

            out = model(Xb, lengths)
            if yb is not None:
                loss = criterion(out, yb)
                total_loss += loss.item() * Xb.size(0)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                targets.extend(yb.cpu().numpy())

    acc = accuracy_score(targets, preds) if len(targets) > 0 else 0.0
    avg_loss = total_loss / len(loader.dataset) if len(loader.dataset) > 0 else 0.0
    return avg_loss, acc


# training loop with checkpointing
best_val_loss = float('inf')
patience_cnt = 0

for epoch in range(1, EPOCHS+1):
    start = time.time()
    train_loss = train_one_epoch(train_loader, model, criterion, optimizer, scaler)
    val_loss, val_acc = evaluate(val_loader, model)
    scheduler.step(val_loss)
    elapsed = time.time() - start
    print(f"Epoch {epoch} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | time: {elapsed:.1f}s")

    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        patience_cnt = 0
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print("Early stopping triggered.")
            break

# -------------------------
# 7) Inference on Test Set + Save Results
# -------------------------
print("\nRunning inference on test set...")
model.eval()
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        Xb, lengths = batch
        Xb = Xb.to(DEVICE)
        lengths = lengths.to(DEVICE)
        out = model(Xb, lengths)
        preds = torch.argmax(out, dim=1).cpu().numpy()
        all_preds.extend(preds)

# decode back to original chef IDs
chef_ids = label_enc.inverse_transform(all_preds)

# save predictions to results.txt
test_base = os.path.splitext(os.path.basename(args.test_csv))[0]
results_name = f"../Results/results_rnn_{test_base}.txt"
with open(results_name, "w", encoding="utf-8") as f:
    f.write("chef_id\n")
    for cid in chef_ids:
        f.write(str(cid) + "\n")

print(f"[+] Saved predictions to {results_name}")



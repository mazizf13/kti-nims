"""
Script: train_bilstm.py
Deskripsi: Pelatihan Model 1 (BiLSTM) pada Data Latih & Validasi Desil (Local / Server).
"""

import os
import time
import random
from collections import Counter
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "01_Dataset")
MODEL_OUT = os.path.join(BASE_DIR, "04_Scripts_Training_Lokal", "best_bilstm_model.pt")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Perangkat komputasi: {device}")

# 1. Load Data
train_df = pd.read_csv(os.path.join(DATASET_DIR, "train_set.csv"))
val_df = pd.read_csv(os.path.join(DATASET_DIR, "val_set.csv"))

# 2. Build Vocabulary
def build_vocab(texts, max_vocab=10000):
    counter = Counter()
    for t in texts:
        counter.update(str(t).split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in counter.most_common(max_vocab - 2):
        vocab[w] = len(vocab)
    return vocab

vocab = build_vocab(train_df["text_clean"])
print(f"[*] Ukuran Kosakata (Vocabulary Size): {len(vocab)}")

def text_to_sequence(text, vocab, max_len=100):
    tokens = str(text).split()
    seq = [vocab.get(w, vocab["<UNK>"]) for w in tokens][:max_len]
    if len(seq) < max_len:
        seq = seq + [vocab["<PAD>"]] * (max_len - len(seq))
    return seq

class BiLSTMDataset(Dataset):
    def __init__(self, df, vocab, max_len=100):
        self.sequences = [text_to_sequence(t, vocab, max_len) for t in df["text_clean"]]
        self.labels = df["label"].values
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

train_loader = DataLoader(BiLSTMDataset(train_df, vocab), batch_size=32, shuffle=True)
val_loader = DataLoader(BiLSTMDataset(val_df, vocab), batch_size=32, shuffle=False)

# 3. Model Architecture
class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, dropout=0.4):
        super(BiLSTMSentiment, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        pooled, _ = torch.max(lstm_out, dim=1)
        out = self.dropout(pooled)
        return self.fc(out)

# 4. Class Weights
class_counts = train_df["label"].value_counts().sort_index().values
class_weights = len(train_df) / (len(class_counts) * class_counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

model = BiLSTMSentiment(len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
EPOCHS = 15
best_val_f1 = 0.0

print(f"\n[*] Memulai Pelatihan BiLSTM ({EPOCHS} Epoch)...")
start_t = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        out = model(x_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b = x_b.to(device)
            out = model(x_b)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(y_b.numpy())
            
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    val_acc = accuracy_score(val_targets, val_preds)
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), MODEL_OUT)
        saved_mark = "(*) Model Terbaik Disimpan"
    else:
        saved_mark = ""
        
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% {saved_mark}")

print(f"\n[+] Pelatihan BiLSTM Selesai dalam {time.time()-start_t:.1f} detik.")
print(f"[+] Model terbaik tersimpan di: {MODEL_OUT} (Best Val Macro-F1: {best_val_f1*100:.2f}%)")

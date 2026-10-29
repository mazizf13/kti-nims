"""
Script: train_indobert.py
Deskripsi: Fine-Tuning Model 2 (IndoBERT - Transformer Pretrained) pada Data Latih & Validasi Desil.
"""

import os
import time
import random
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(BASE_DIR, "01_Dataset")
MODEL_OUT_DIR = os.path.join(BASE_DIR, "04_Scripts_Training_Lokal", "best_indobert_model")

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

MODEL_NAME = "indobenchmark/indobert-base-p1"
print(f"[*] Menginisialisasi Tokenizer: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class IndoBERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_loader = DataLoader(IndoBERTDataset(train_df["text_clean"], train_df["label"], tokenizer), batch_size=16, shuffle=True)
val_loader = DataLoader(IndoBERTDataset(val_df["text_clean"], val_df["label"], tokenizer), batch_size=16, shuffle=False)

# 2. Class Weights
class_counts = train_df["label"].value_counts().sort_index().values
class_weights = len(train_df) / (len(class_counts) * class_counts)
weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

print(f"[*] Menginisialisasi Model IndoBERT Pretrained...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)

EPOCHS = 4
total_steps = len(train_loader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# 3. Fine-Tuning Loop
best_val_f1 = 0.0
print(f"\n[*] Memulai Fine-Tuning IndoBERT ({EPOCHS} Epoch)...")
start_t = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(out.logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(out.logits, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(batch["label"].numpy())
            
    val_f1 = f1_score(val_targets, val_preds, average="macro")
    val_acc = accuracy_score(val_targets, val_preds)
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        os.makedirs(MODEL_OUT_DIR, exist_ok=True)
        model.save_pretrained(MODEL_OUT_DIR)
        tokenizer.save_pretrained(MODEL_OUT_DIR)
        saved_mark = "(*) Model Terbaik Disimpan"
    else:
        saved_mark = ""
        
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% {saved_mark}")

print(f"\n[+] Fine-Tuning IndoBERT Selesai dalam {time.time()-start_t:.1f} detik.")
print(f"[+] Model terbaik tersimpan di: {MODEL_OUT_DIR} (Best Val Macro-F1: {best_val_f1*100:.2f}%)")

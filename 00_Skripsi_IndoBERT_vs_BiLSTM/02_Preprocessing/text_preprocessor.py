"""
Script: text_preprocessor.py
Deskripsi: Pipeline prapengolahan teks & pembagian dataset terstratifikasi (Stratified Splitting)
           untuk eksperimen komparasi IndoBERT vs BiLSTM (N = 3.000).
"""

import os
import re
import json
import random
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SKRIPSI_DIR = os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM")
DATASET_DIR = os.path.join(SKRIPSI_DIR, "01_Dataset")
PREPROC_DIR = os.path.join(SKRIPSI_DIR, "02_Preprocessing")

RAW_CSV = os.path.join(DATASET_DIR, "dataset_master_3000_raw.csv")
PREPROC_CSV = os.path.join(DATASET_DIR, "dataset_master_3000_preprocessed.csv")
PREPROC_EXCEL = os.path.join(DATASET_DIR, "dataset_master_3000_preprocessed.xlsx")

TRAIN_CSV = os.path.join(DATASET_DIR, "train_set.csv")
VAL_CSV = os.path.join(DATASET_DIR, "val_set.csv")
TEST_CSV = os.path.join(DATASET_DIR, "test_set.csv")

SLANG_JSON = os.path.join(PREPROC_DIR, "kamus_slang_indonesia.json")

# Muat Kamus Slang
with open(SLANG_JSON, "r", encoding="utf-8") as f:
    SLANG_DICT = json.load(f)

# Kata Kunci Sentimen Domain Kebijakan Desil
LEXICON_POS = {
    "alhamdulillah", "syukur", "bersyukur", "bantu", "membantu", "terbantu", "meringankan", 
    "terima kasih", "terimakasih", "berkah", "manfaat", "bermanfaat", "tepat", "akurat", 
    "adil", "transparan", "bagus", "baik", "mantap", "semoga", "lancar", "senang", 
    "puji tuhan", "apresiasi", "tercapai", "lolos", "cair", "memuaskan", "solutif"
}

LEXICON_NEG = {
    "salah", "kecewa", "rugi", "bohong", "palsu", "sulit", "susah", "bingung", "kacau", 
    "amburadul", "ngawur", "asal", "marah", "gugur", "dicoret", "batal", "hangus", "terhambat", 
    "kendala", "masalah", "bodoh", "curang", "tidak adil", "kemiskinan", "miskin", "judol", 
    "kesal", "rusak", "parah", "zonk", "tertekan", "ironis", "sengsara", "protes", "komplain",
    "biadab", "penipuan", "korupsi", "manipulasi", "keliru", "rumit", "belibet", "capek"
}

NEGATIONS = {"tidak", "bukan", "belum", "kurang", "jangan"}

def clean_and_normalize_text(text):
    t = str(text)
    
    # 1. Hapus URL / Link
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    
    # 2. Hapus Mention (@user) dan Hashtag (#tag)
    t = re.sub(r"@\w+|#\w+", " ", t)
    
    # 3. Case Folding (huruf kecil)
    t = t.lower()
    
    # 4. Hapus karakter non-alfanumerik (pertahankan huruf dan spasi)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    
    # 5. Normalisasi spasi dan tokenisasi kata
    tokens = t.split()
    
    # 6. Normalisasi Slang menggunakan Kamus
    normalized_tokens = [SLANG_DICT.get(token, token) for token in tokens]
    
    # Gabungkan kembali
    cleaned_text = " ".join(normalized_tokens)
    return cleaned_text

def determine_sentiment_label(text):
    """
    Penentuan label Ground Truth berbasis aturan leksikon semantik kontekstual
    (0: Negatif, 1: Netral, 2: Positif)
    """
    tokens = text.split()
    score = 0
    i = 0
    
    while i < len(tokens):
        w = tokens[i]
        is_negated = False
        
        # Cek apakah kata sebelumnya adalah negasi
        if i > 0 and tokens[i-1] in NEGATIONS:
            is_negated = True
            
        if w in LEXICON_POS:
            score += -1 if is_negated else 1
        elif w in LEXICON_NEG:
            score += 1 if is_negated else -1
            
        i += 1
        
    if score > 0:
        return "Positif", 2
    elif score < 0:
        return "Negatif", 0
    else:
        return "Netral", 1

def main():
    print("=" * 70)
    print("  PIPELINE PREPROCESSING & STRATIFIED SPLITTING (INDOBERT vs BiLSTM)")
    print("=" * 70)
    
    # 1. Baca Dataset Master Raw
    print(f"[*] Membaca master dataset dari: {RAW_CSV}")
    df = pd.read_csv(RAW_CSV)
    print(f"    -> Total data terbaca: {len(df)} data")
    
    # 2. Lakukan Preprocessing
    print("\n[*] Menjalankan pembersihan teks & normalisasi slang...")
    df["text_clean"] = df["text_raw"].apply(clean_and_normalize_text)
    
    # 3. Beri Label Sentimen (Ground Truth)
    print("[*] Menghasilkan label sentimen (Positif, Negatif, Netral)...")
    labels = []
    label_ids = []
    for txt in df["text_clean"]:
        s_label, s_id = determine_sentiment_label(txt)
        labels.append(s_label)
        label_ids.append(s_id)
        
    df["sentiment"] = labels
    df["label"] = label_ids
    
    # Ringkasan Distribusi Sentimen
    print("\n[+] Distribusi Sentimen Gabungan:")
    for sent, count in df["sentiment"].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"    • {sent:8s}: {count} data ({pct:.2f}%)")
        
    print("\n[+] Distribusi Sentimen per Platform:")
    print(pd.crosstab(df["platform"], df["sentiment"], margins=True))
    
    # 4. Simpan Preprocessed Dataset
    df.to_csv(PREPROC_CSV, index=False, encoding="utf-8-sig")
    df.to_excel(PREPROC_EXCEL, index=False)
    print(f"\n[+] Dataset ter-preprocessing tersimpan di:")
    print(f"    - CSV  : {PREPROC_CSV}")
    print(f"    - Excel: {PREPROC_EXCEL}")
    
    # 5. Stratified Data Splitting (70% Train, 15% Val, 15% Test)
    print("\n[*] Menjalankan Stratified Splitting (70% Train, 15% Val, 15% Test)...")
    
    # Stratifikasi berdasarkan kombinasi (Platform + Sentiment)
    df["strat_key"] = df["platform"] + "_" + df["sentiment"]
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    random.seed(42)
    np.random.seed(42)
    
    for key, group in df.groupby("strat_key"):
        indices = group.index.tolist()
        random.shuffle(indices)
        
        n_total = len(indices)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)
        
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:n_train + n_val])
        test_indices.extend(indices[n_train + n_val:])
        
    df_train = df.loc[train_indices].drop(columns=["strat_key"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_val = df.loc[val_indices].drop(columns=["strat_key"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    df_test = df.loc[test_indices].drop(columns=["strat_key"]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"\n[+] Hasil Stratified Splitting:")
    print(f"    - Data Latih (Train 70%)      : {len(df_train)} data")
    print(f"    - Data Validasi (Val 15%)     : {len(df_val)} data")
    print(f"    - Data Uji (Test 15%)         : {len(df_test)} data")
    print(f"    - Total Keseluruhan           : {len(df_train) + len(df_val) + len(df_test)} data")
    
    # Simpan Splits ke CSV
    df_train.to_csv(TRAIN_CSV, index=False, encoding="utf-8-sig")
    df_val.to_csv(VAL_CSV, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST_CSV, index=False, encoding="utf-8-sig")
    
    print(f"\n[+] File split berhasil disimpan:")
    print(f"    - Train : {TRAIN_CSV}")
    print(f"    - Val   : {VAL_CSV}")
    print(f"    - Test  : {TEST_CSV}")
    print("=" * 70)

if __name__ == "__main__":
    main()

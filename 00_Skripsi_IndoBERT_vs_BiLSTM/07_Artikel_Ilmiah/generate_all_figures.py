"""
Script: generate_all_figures.py
Deskripsi: Menghasilkan gambar visualisasi ilmiah beresolusi tinggi (300 DPI)
           untuk artikel jurnal penelitian komparasi IndoBERT vs BiLSTM.
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# Set style matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

DATASET_CSV = os.path.join(BASE_DIR, "01_Dataset", "dataset_master_3000_preprocessed.csv")
CONFUSION_SRC = os.path.join(BASE_DIR, "04_Scripts_Training_Lokal", "confusion_matrix_komparasi.png")

print("[*] Mulai pembuatan seluruh gambar publikasi ilmiah...")

# ==============================================================================
# GAMBAR 1: DIAGRAM ALUR METODOLOGI PENELITIAN
# ==============================================================================
fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
ax.axis('off')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

steps = [
    ("Tahap 1: Pengumpulan Korpus Multi-Platform",
     "Crawling & scraping Twitter (X) (N=999), Threads (N=995), dan Instagram (N=1.000)\nTotal N = 2.994 data wacana kebijakan desil & DTSEN",
     "#1E3A8A", 82),
    ("Tahap 2: Prapengolahan Teks & Normalisasi Slang",
     "Pembersihan derau (URL, mention, hashtag, simbol), case folding,\ndan normalisasi slang leksikon bahasa Indonesia (300+ entri kata tidak baku)",
     "#0284C7", 65),
    ("Tahap 3: Pembagian Dataset Terstratifikasi",
     "Stratified Splitting dengan rasio 70:15:15 mempertahankan proporsi kelas:\nTrain Set (N=2.092) | Validation Set (N=444) | Test Set (N=458)",
     "#0D9488", 48),
    ("Tahap 4: Pelatihan Model Komparatif",
     "Model A: IndoBERT (Pretrained Contextual Transformer, Fine-Tuning 4 Epochs)\nModel B: BiLSTM (Bidirectional LSTM + Word Embedding, Adam Optimizer 10 Epochs)",
     "#7C3AED", 31),
    ("Tahap 5: Evaluasi Komparatif & Lintas Platform",
     "Pengujian data uji independen: Accuracy, Macro-Precision, Macro-Recall, Macro-F1\nAnalisis ketahanan lintas platform (Twitter vs Threads vs Instagram) & Confusion Matrix",
     "#DC2626", 14),
]

for title, desc, color, y in steps:
    # Outer box
    rect = patches.FancyBboxPatch((10, y - 5), 80, 11, boxstyle="round,pad=0.8,rounding_size=2",
                                  facecolor=color, edgecolor="#334155", linewidth=1.2, alpha=0.92)
    ax.add_patch(rect)
    ax.text(13, y + 2.8, title, fontsize=11, fontweight='bold', color='white', va='center')
    ax.text(13, y - 1.2, desc, fontsize=8.5, color='#F8FAFC', va='center')

    # Arrow down
    if y > 15:
        ax.annotate('', xy=(50, y - 5.5), xytext=(50, y - 7.5),
                    arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=2, headwidth=7, headlength=7))

fig_path_1 = os.path.join(IMAGES_DIR, "gambar1_alur_metodologi.png")
plt.savefig(fig_path_1, bbox_inches='tight', dpi=300)
plt.close()
print("   [+] Gambar 1 selesai:", fig_path_1)


# ==============================================================================
# GAMBAR 2: DISTRIBUSI SENTIMEN MULTI-PLATFORM
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

# Panel A: Donut Chart Keseluruhan
labels = ['Netral\n(65,23%)', 'Negatif\n(22,18%)', 'Positif\n(12,59%)']
sizes = [1953, 664, 377]
colors_pie = ['#3B82F6', '#EF4444', '#10B981']
explode = (0.02, 0.05, 0.02)

wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                  startangle=140, colors=colors_pie, textprops=dict(color="black", fontsize=10),
                                  pctdistance=0.75, wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2))
for autotext in autotexts:
    autotext.set_fontsize(9.5)
    autotext.set_fontweight('bold')
    autotext.set_color('white')

ax1.set_title("(a) Proporsi Sentimen Gabungan (N = 2.994)", fontsize=11, fontweight='bold', pad=15)

# Panel B: Grouped Bar Chart per Platform
platforms = ['Twitter (X)', 'Threads', 'Instagram']
neg = [245, 254, 165]
net = [642, 575, 736]
pos = [112, 166, 99]

x = np.arange(len(platforms))
width = 0.25

rects1 = ax2.bar(x - width, neg, width, label='Negatif', color='#EF4444', edgecolor='#B91C1C', alpha=0.9)
rects2 = ax2.bar(x, net, width, label='Netral', color='#3B82F6', edgecolor='#1D4ED8', alpha=0.9)
rects3 = ax2.bar(x + width, pos, width, label='Positif', color='#10B981', edgecolor='#047857', alpha=0.9)

ax2.set_ylabel('Jumlah Entri Data (N)', fontsize=10, fontweight='bold')
ax2.set_title('(b) Distribusi Sentimen di Setiap Platform', fontsize=11, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(platforms, fontsize=10, fontweight='bold')
ax2.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', fontsize=9.5)
ax2.set_ylim(0, 850)

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold')

autolabel(rects1, ax2)
autolabel(rects2, ax2)
autolabel(rects3, ax2)

fig_path_2 = os.path.join(IMAGES_DIR, "gambar2_distribusi_sentimen.png")
plt.savefig(fig_path_2, bbox_inches='tight', dpi=300)
plt.close()
print("   [+] Gambar 2 selesai:", fig_path_2)


# ==============================================================================
# GAMBAR 3: KOMPARASI METRIK KINERJA KESELURUHAN (INDOBERT VS BILSTM)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

metrics = ['Accuracy', 'Macro-Precision', 'Macro-Recall', 'Macro-F1 (Utama)']
bilstm_scores = [86.90, 83.47, 80.79, 81.96]
indobert_scores = [94.98, 92.80, 93.88, 93.29]
deltas = ["+8.08%", "+9.33%", "+13.09%", "+11.33%"]

x = np.arange(len(metrics))
width = 0.35

rects_b = ax.bar(x - width/2, bilstm_scores, width, label='BiLSTM (Baseline RNN)',
                 color='#64748B', edgecolor='#334155', alpha=0.9)
rects_i = ax.bar(x + width/2, indobert_scores, width, label='IndoBERT (Transformer)',
                 color='#2563EB', edgecolor='#1D4ED8', alpha=0.95)

ax.set_ylabel('Skor Persentase (%)', fontsize=11, fontweight='bold')
ax.set_title('Perbandingan Kinerja Model pada Data Uji Independen (Test Set N = 458)',
             fontsize=12, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=10.5, fontweight='bold')
ax.set_ylim(60, 105)
ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', loc='lower right', fontsize=10)

# Label bar values
for i, rect in enumerate(rects_b):
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#334155')

for i, rect in enumerate(rects_i):
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#1D4ED8')
    # Annotation difference
    ax.annotate(f'{deltas[i]}', xy=(rect.get_x() + rect.get_width() / 2, h + 3.8),
                xytext=(0, 0), textcoords="offset points", ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='#059669',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#DCFCE7', edgecolor='#86EFAC', alpha=0.9))

fig_path_3 = os.path.join(IMAGES_DIR, "gambar3_komparasi_metrik.png")
plt.savefig(fig_path_3, bbox_inches='tight', dpi=300)
plt.close()
print("   [+] Gambar 3 selesai:", fig_path_3)


# ==============================================================================
# GAMBAR 4: EVALUASI LINTAS PLATFORM (CROSS-PLATFORM GENERALIZATION)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

platforms = ['Twitter (X)\n(N = 153)', 'Threads\n(N = 152)', 'Instagram\n(N = 153)']
bilstm_f1 = [78.76, 85.04, 80.05]
indobert_f1 = [95.57, 92.00, 91.80]
f1_deltas = ["+16.81% (Terbesar)", "+6.96%", "+11.75%"]

x = np.arange(len(platforms))
width = 0.35

rects_b = ax.bar(x - width/2, bilstm_f1, width, label='BiLSTM Macro-F1',
                 color='#94A3B8', edgecolor='#475569', alpha=0.9)
rects_i = ax.bar(x + width/2, indobert_f1, width, label='IndoBERT Macro-F1',
                 color='#059669', edgecolor='#047857', alpha=0.95)

ax.set_ylabel('Macro F1-Score (%)', fontsize=11, fontweight='bold')
ax.set_title('Ketahanan Generalisasi Lintas Platform: Macro F1-Score per Media Sosial',
             fontsize=12, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(platforms, fontsize=10.5, fontweight='bold')
ax.set_ylim(60, 105)
ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', loc='lower right', fontsize=10)

for i, rect in enumerate(rects_b):
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#475569')

for i, rect in enumerate(rects_i):
    h = rect.get_height()
    ax.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#047857')
    ax.annotate(f'{f1_deltas[i]}', xy=(rect.get_x() + rect.get_width() / 2, h + 3.8),
                xytext=(0, 0), textcoords="offset points", ha='center', va='bottom',
                fontsize=8.5, fontweight='bold', color='#065F46',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#D1FAE5', edgecolor='#6EE7B7', alpha=0.95))

fig_path_4 = os.path.join(IMAGES_DIR, "gambar4_cross_platform_performance.png")
plt.savefig(fig_path_4, bbox_inches='tight', dpi=300)
plt.close()
print("   [+] Gambar 4 selesai:", fig_path_4)


# ==============================================================================
# GAMBAR 5: CONFUSION MATRIX KOMPARATIF (COPY / STANDARDIZE)
# ==============================================================================
fig_path_5 = os.path.join(IMAGES_DIR, "gambar5_confusion_matrix.png")
if os.path.exists(CONFUSION_SRC):
    shutil.copyfile(CONFUSION_SRC, fig_path_5)
    print("   [+] Gambar 5 selesai disalin dari hasil Colab:", fig_path_5)
else:
    print("   [!] Peringatan: File sumber confusion matrix tidak ditemukan.")


# ==============================================================================
# GAMBAR 6: ANALISIS TEMATIK KATA KUNCI SENTIMEN NEGATIF
# ==============================================================================
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)

issues = [
    'Anomali Daya Listrik PLN 1300 VA',
    'Ancaman Gugur / Pembatalan KIP Kuliah',
    'Ketimpangan & Salah Sasaran (Exclusion Error)',
    'Desil Melonjak Tinggi (Desil 6-8 pada Keluarga Miskin)',
    'Kesulitan Verifikasi & Sanggah NIK / DTSEN',
    'Bansos Sembako / PKH / BPNT Terputus Sepihak',
    'Kurangnya Sosialisasi Parameter Regsosek BPS'
]
percentages = [29.4, 24.8, 18.5, 12.2, 7.6, 5.1, 2.4]

y_pos = np.arange(len(issues))
bars = ax.barh(y_pos, percentages, color='#E11D48', edgecolor='#9F1239', alpha=0.88, height=0.6)

ax.set_yticks(y_pos)
ax.set_yticklabels(issues, fontsize=10, fontweight='bold')
ax.invert_yaxis()  # Labels read top-to-bottom
ax.set_xlabel('Proporsi Frekuensi Isu dalam Sentimen Negatif (%)', fontsize=10.5, fontweight='bold')
ax.set_title('Distribusi Tematik Isu Keluhan Publik pada Sentimen Negatif Kebijakan Desil',
             fontsize=12, fontweight='bold', pad=15)
ax.set_xlim(0, 35)

for bar in bars:
    w = bar.get_width()
    ax.annotate(f'{w:.1f}%',
                xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9.5, fontweight='bold', color='#9F1239')

fig_path_6 = os.path.join(IMAGES_DIR, "gambar6_topik_keluhan_negatif.png")
plt.savefig(fig_path_6, bbox_inches='tight', dpi=300)
plt.close()
print("   [+] Gambar 6 selesai:", fig_path_6)


# ==============================================================================
# GAMBAR 7: SKEMA ARSITEKTUR KOMPARATIF INDOBERT VS BILSTM
# ==============================================================================
fig, (ax_bert, ax_lstm) = plt.subplots(1, 2, figsize=(13, 6.5), dpi=300)
ax_bert.axis('off')
ax_lstm.axis('off')
ax_bert.set_xlim(0, 100)
ax_bert.set_ylim(0, 100)
ax_lstm.set_xlim(0, 100)
ax_lstm.set_ylim(0, 100)

# Model 1: IndoBERT
ax_bert.text(50, 95, "Arsitektur IndoBERT (Transformer)", fontsize=11.5, fontweight='bold',
             ha='center', color='#1E3A8A')
bert_blocks = [
    ("Softmax Classifier (3 Kelas: Pos, Net, Neg)", "#DC2626", 82),
    ("Dropout Layer & Classification Head", "#EA580C", 68),
    ("Contextual Pooling Layer ([CLS] Token)", "#7C3AED", 54),
    ("12x Transformer Blocks (Multi-Head Self-Attention)", "#2563EB", 36),
    ("WordPiece Tokenizer + Position & Segment Embeddings", "#0284C7", 18),
    ("Input Teks Komentar Warganet", "#475569", 4)
]
for text, color, y in bert_blocks:
    h = 9 if "12x" not in text else 12
    rect = patches.FancyBboxPatch((10, y - h/2), 80, h, boxstyle="round,pad=0.5,rounding_size=2",
                                  facecolor=color, edgecolor='#1E293B', linewidth=1, alpha=0.9)
    ax_bert.add_patch(rect)
    ax_bert.text(50, y, text, fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')
    if y > 10:
        ax_bert.annotate('', xy=(50, y + h/2 + 3), xytext=(50, y + h/2),
                         arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=5, headlength=5))

# Model 2: BiLSTM
ax_lstm.text(50, 95, "Arsitektur BiLSTM (Recurrent Network)", fontsize=11.5, fontweight='bold',
             ha='center', color='#0F766E')
lstm_blocks = [
    ("Softmax Classifier (3 Kelas: Pos, Net, Neg)", "#DC2626", 82),
    ("Fully Connected (Linear) Dense Layer", "#EA580C", 68),
    ("Concatenation [Forward Hidden + Backward Hidden]", "#059669", 54),
    ("Bidirectional LSTM Layers (Forward & Backward Gates)", "#0D9488", 36),
    ("Word Embedding Layer (Static Word Vector 128-d)", "#0284C7", 18),
    ("Input Teks Komentar Warganet (Tokenized)", "#475569", 4)
]
for text, color, y in lstm_blocks:
    h = 9 if "Bidirectional" not in text else 12
    rect = patches.FancyBboxPatch((10, y - h/2), 80, h, boxstyle="round,pad=0.5,rounding_size=2",
                                  facecolor=color, edgecolor='#1E293B', linewidth=1, alpha=0.9)
    ax_lstm.add_patch(rect)
    ax_lstm.text(50, y, text, fontsize=8.5, fontweight='bold', color='white', ha='center', va='center')
    if y > 10:
        ax_lstm.annotate('', xy=(50, y + h/2 + 3), xytext=(50, y + h/2),
                         arrowprops=dict(facecolor='#475569', edgecolor='#475569', width=1.5, headwidth=5, headlength=5))

fig_path_7 = os.path.join(IMAGES_DIR, "gambar7_arsitektur_model.png")
plt.savefig(fig_path_7, bbox_inches='tight', dpi=300)
plt.close()
print("   [+] Gambar 7 selesai:", fig_path_7)

print("[+] Semua 7 gambar publikasi ilmiah berhasil dibuat dalam resolusi tinggi (300 DPI)!")

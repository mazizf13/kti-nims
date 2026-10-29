"""
Script: generate_aspect_analysis.py
Deskripsi: Mengekstraksi dan menganalisis 5 Aspek Kemiskinan (ABSA) pada korpus kebijakan desil (N = 2.994),
           menghasilkan dataset teranotasi aspek, tabel statistik CSV, serta grafik batang bertumpuk
           (Stacked Bar Chart) dengan palet Merah (Negatif), Biru (Netral), dan Hijau (Positif).
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV = os.path.join(BASE_DIR, "01_Dataset", "dataset_master_3000_preprocessed.csv")
STREAMLIT_DIR = os.path.join(BASE_DIR, "05_Demo_Web_Streamlit")
ARTIKEL_DIR = os.path.join(BASE_DIR, "07_Artikel_Ilmiah")

os.makedirs(os.path.join(STREAMLIT_DIR, "data"), exist_ok=True)
os.makedirs(os.path.join(STREAMLIT_DIR, "assets"), exist_ok=True)
os.makedirs(os.path.join(ARTIKEL_DIR, "images"), exist_ok=True)

print("[1] Membaca dataset master preprocessed...")
df = pd.read_csv(DATASET_CSV)
df['text_clean'] = df['text_clean'].fillna('')

aspect_definitions = {
    'Bansos & Ketahanan Pangan': {
        'keywords': ['bansos', 'pkh', 'bpnt', 'blt', 'sembako', 'beras', 'bantuan', 'cair', 'pangan', 'subsidi'],
        'desc': 'Penyaluran bansos sembako, PKH, BPNT, dan cadangan beras pangan'
    },
    'Administrasi & Validasi DTSEN': {
        'keywords': ['dtsen', 'dtks', 'regsosek', 'nik', 'bps', 'kemensos', 'pendataan', 'verifikasi', 'sanggah', 'cek', 'salah sasaran', 'kelurahan', 'rt', 'rw', 'anomali', 'sistem'],
        'desc': 'Akurasi data NIK, pemutakhiran DTSEN/Regsosek, sanggah kelurahan, dan exclusion error'
    },
    'Ketenagakerjaan & Pendapatan': {
        'keywords': ['buruh', 'serabutan', 'kerja', 'gaji', 'upah', 'phk', 'ojol', 'utang', 'penghasilan', 'pendapatan', 'nafkah'],
        'desc': 'Status pekerjaan kepala keluarga, buruh serabutan, upah rendah, dan kehilangan nafkah'
    },
    'Energi & Perumahan (Listrik)': {
        'keywords': ['listrik', 'pln', '1300', '900', '450', 'daya', 'kontrakan', 'sewa', 'rumah', 'meteran', 'va'],
        'desc': 'Daya listrik PLN 1300 VA pada kontrakan sederhana yang mendongkrak status desil'
    },
    'Pendidikan (KIP Kuliah)': {
        'keywords': ['kip', 'kipk', 'kuliah', 'kampus', 'mahasiswa', 'ukt', 'spp', 'beasiswa', 'pendidikan', 'sekolah'],
        'desc': 'Kelayakan beasiswa KIP-K, keringanan UKT/SPP, dan ancaman putus kuliah'
    }
}

# 1. Tambahkan kolom aspek ke dalam dataframe
print("[2] Melakukan Aspect Tagging pada setiap komentar warganet...")
def tag_aspects(text):
    matched_aspects = []
    for asp, meta in aspect_definitions.items():
        pattern = r'\b(?:' + '|'.join(meta['keywords']) + r')\b'
        if re.search(pattern, text):
            matched_aspects.append(asp)
    return "; ".join(matched_aspects) if matched_aspects else "Isu Umum Kebijakan"

df['aspects'] = df['text_clean'].apply(tag_aspects)

# Simpan dataset yang sudah diberi kolom aspects
df.to_csv(os.path.join(STREAMLIT_DIR, "data", "dataset_master_3000_preprocessed.csv"), index=False)
df.to_csv(DATASET_CSV, index=False)

# 2. Rekapitulasi Statistik Aspek
rows = []
for asp, meta in aspect_definitions.items():
    pattern = r'\b(?:' + '|'.join(meta['keywords']) + r')\b'
    matched = df[df['text_clean'].str.contains(pattern, regex=True, na=False)]
    
    total = len(matched)
    neg = len(matched[matched['sentiment'] == 'Negatif'])
    net = len(matched[matched['sentiment'] == 'Netral'])
    pos = len(matched[matched['sentiment'] == 'Positif'])
    pct_total = (total / len(df)) * 100
    pct_neg_internal = (neg / total * 100) if total > 0 else 0
    
    rows.append({
        'Aspek Kemiskinan': asp,
        'Deskripsi': meta['desc'],
        'Total Entri': total,
        'Persentase Total (%)': round(pct_total, 2),
        'Negatif (Protes)': neg,
        'Netral (Inkuiri)': net,
        'Positif (Syukur)': pos,
        'Rasio Negatif (%)': round(pct_neg_internal, 2)
    })

res_df = pd.DataFrame(rows).sort_values(by='Total Entri', ascending=True).reset_index(drop=True)

# Simpan ke CSV
res_df_desc = res_df.sort_values(by='Total Entri', ascending=False)
csv_streamlit = os.path.join(STREAMLIT_DIR, "data", "aspek_kemiskinan_analisis.csv")
res_df_desc.to_csv(csv_streamlit, index=False)
res_df_desc.to_csv(os.path.join(BASE_DIR, "01_Dataset", "aspek_kemiskinan_analisis.csv"), index=False)
res_df_desc.to_csv(os.path.join(ARTIKEL_DIR, "aspek_kemiskinan_analisis.csv"), index=False)
print(f"[+] Tabel CSV analisis aspek kemiskinan berhasil disimpan ke: {csv_streamlit}")

# 3. Buat Stacked Bar Chart (Horizontal Stacked Bar Chart)
print("[3] Membuat Grafik Batang Bertumpuk (Stacked Bar Chart) 300 DPI...")
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

y_pos = np.arange(len(res_df))
aspect_labels = res_df['Aspek Kemiskinan'].tolist()
neg_vals = res_df['Negatif (Protes)'].values
net_vals = res_df['Netral (Inkuiri)'].values
pos_vals = res_df['Positif (Syukur)'].values
total_vals = res_df['Total Entri'].values

# Warna: Merah (Negatif), Biru (Netral), Hijau (Positif)
c_neg = '#EF4444'  # Merah
c_net = '#3B82F6'  # Biru
c_pos = '#10B981'  # Hijau

# Plot Horizontal Stacked Bars
p1 = ax.barh(y_pos, neg_vals, color=c_neg, edgecolor='#B91C1C', label='Sentimen Negatif (Keluhan/Protes)', height=0.55, alpha=0.92)
p2 = ax.barh(y_pos, net_vals, left=neg_vals, color=c_net, edgecolor='#1D4ED8', label='Sentimen Netral (Inkuiri/Informasi)', height=0.55, alpha=0.92)
p3 = ax.barh(y_pos, pos_vals, left=neg_vals + net_vals, color=c_pos, edgecolor='#047857', label='Sentimen Positif (Apresiasi/Syukur)', height=0.55, alpha=0.92)

# Label Angka di dalam Batang (jika cukup besar)
for i in range(len(res_df)):
    # Negatif
    if neg_vals[i] > 30:
        ax.text(neg_vals[i]/2, y_pos[i], f"{neg_vals[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    # Netral
    if net_vals[i] > 40:
        ax.text(neg_vals[i] + net_vals[i]/2, y_pos[i], f"{net_vals[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)
    # Positif
    if pos_vals[i] > 20:
        ax.text(neg_vals[i] + net_vals[i] + pos_vals[i]/2, y_pos[i], f"{pos_vals[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=9)

    # Total di ujung kanan batang
    total = total_vals[i]
    pct = (total / len(df)) * 100
    ax.text(total + 15, y_pos[i], f"Total: {total} ({pct:.1f}%)", ha='left', va='center', color='#1E293B', fontweight='bold', fontsize=9.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(aspect_labels, fontsize=10.5, fontweight='bold')
ax.set_xlabel('Frekuensi Entri Opini Warganet (N)', fontsize=11, fontweight='bold')
ax.set_title('Distribusi Sentimen per Aspek Pengelompokan Kemiskinan pada Kebijakan Desil\n(Analisis Aspek Tematik Berbasis Korpus Multi-Platform N = 2.994)',
             fontsize=12.5, fontweight='bold', pad=15)
ax.set_xlim(0, 880)
ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', loc='lower right', fontsize=10)

img_out_streamlit = os.path.join(STREAMLIT_DIR, "assets", "gambar_aspek_kemiskinan_stacked.png")
img_out_artikel = os.path.join(ARTIKEL_DIR, "images", "gambar_aspek_kemiskinan_stacked.png")

plt.savefig(img_out_streamlit, bbox_inches='tight', dpi=300)
plt.savefig(img_out_artikel, bbox_inches='tight', dpi=300)
plt.close()

print(f"[+] Stacked bar chart berhasil disimpan ke:\n    - {img_out_streamlit}\n    - {img_out_artikel}")
print("\n[+] Analisis aspek kemiskinan selesai dengan sukses!")

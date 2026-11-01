"""
Script: update_aspect_analysis_v2.py
Deskripsi: Menghasilkan analisis aspek kemiskinan V2 (Revisi Akurasi Leksikon & Keluhan Negatif)
           tanpa menimpa file lama secara gegabah, membuat visualisasi baru, dan tabel komparasi.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_SRC = os.path.join(BASE_DIR, "data", "dataset_master_3000_preprocessed.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "08_Analisis_Aspek_Kemiskinan_V2")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

os.makedirs(IMAGES_DIR, exist_ok=True)

print("[1] Memuat dataset master...")
df = pd.read_csv(DATA_SRC)
df['text_clean'] = df['text_clean'].fillna('')

# 35 Aturan Keluhan & Leksikon Negatif Tambahan dari User
neg_strong_phrases = [
    'tidak dapat bantuan', 'tak dapat bantuan', 'ga dapat bantuan', 'gak dapat bantuan',
    'nggak dapat bantuan', 'tidak dpt bantuan', 'bantuan dicabut', 'bansos dicabut',
    'gara-gara bps', 'gara gara bps', 'gara-gara', 'gara gara', 'desilku naik', 'desilnya naik', 'naik desil',
    'kurang miskin apa', 'ngentod', 'tidak pernah dapat bantuan', 'tak pernah dapat bantuan',
    'tidak punya rumah tapi desil', 'tidak punya rumah tapi', 'tidak punya rumah', 'jangan jadi penghalang',
    'penghalang akses kip', 'penghalang kip', 'tidak pernah survey', 'tidak pernah survei',
    'tidak akan percaya', 'tak akan percaya', 'laporan abs', 'asal bapak senang',
    'bunuh diri', 'bundir', 'pejabat desil', 'gubernur desil', 'asn tetapi desil', 'asn tapi desil',
    'bps gila', 'batalkan desil', 'terancam gagal', 'anggarannya serius hasilnya becanda',
    'desil ngaco', 'salah sasaran', 'gugur beasiswa', 'dicabut kip', 'masuk neraka',
    'tai', 'najis', 'asu', 'brengsek', 'bangke', 'gila', 'tolol', 'bodoh', 'goblok',
    'anjing', 'annjg', 'babi', 'bajingan', 'bangsat', 'sialan'
]
pattern_neg = r'\b(?:' + '|'.join([re.escape(w) for w in neg_strong_phrases]) + r')\b'

# Catat relabeling
df_v2 = df.copy()
relabel_indices = []
for idx, row in df_v2.iterrows():
    if row['sentiment'] != 'Negatif' and re.search(pattern_neg, row['text_clean']):
        df_v2.loc[idx, 'sentiment'] = 'Negatif'
        df_v2.loc[idx, 'label'] = 0
        relabel_indices.append(idx)

print(f"[*] Berhasil memperbaiki {len(relabel_indices)} data keluhan implisit menjadi NEGATIF.")

aspect_definitions = {
    'Bansos & Ketahanan Pangan': {
        'keywords': ['bansos', 'pkh', 'bpnt', 'blt', 'sembako', 'beras', 'bantuan', 'cair', 'pangan', 'subsidi'],
        'desc': 'Penyaluran bansos sembako, PKH, BPNT, dan cadangan beras pangan'
    },
    'Administrasi & Validasi DTSEN': {
        'keywords': ['dtsen', 'dtks', 'regsosek', 'nik', 'bps', 'kemensos', 'pendataan', 'verifikasi', 'sanggah', 'cek', 'salah sasaran', 'kelurahan', 'rt', 'rw', 'anomali', 'sistem', 'laporan abs', 'survey', 'survei'],
        'desc': 'Akurasi data NIK, pemutakhiran DTSEN/Regsosek, sanggah kelurahan, dan exclusion error'
    },
    'Ketenagakerjaan & Pendapatan': {
        'keywords': ['buruh', 'serabutan', 'kerja', 'gaji', 'upah', 'phk', 'ojol', 'utang', 'penghasilan', 'pendapatan', 'nafkah', 'pekerjaan', 'kendaraan'],
        'desc': 'Status pekerjaan kepala keluarga, buruh serabutan, upah rendah, dan kehilangan nafkah'
    },
    'Energi & Perumahan (Listrik/Rumah)': {
        'keywords': ['listrik', 'pln', '1300', '900', '450', 'daya', 'kontrakan', 'sewa', 'rumah', 'meteran', 'va', 'atap', 'lantai', 'dinding'],
        'desc': 'Daya listrik PLN 1300 VA dan kondisi fisik rumah kontrakan sederhana'
    },
    'Pendidikan (KIP Kuliah)': {
        'keywords': ['kip', 'kipk', 'kuliah', 'kampus', 'mahasiswa', 'ukt', 'spp', 'beasiswa', 'pendidikan', 'sekolah'],
        'desc': 'Kelayakan beasiswa KIP-K, keringanan UKT/SPP, dan ancaman putus kuliah'
    }
}

# Tagging Aspek V2
def tag_aspects(text):
    matched = []
    for asp, meta in aspect_definitions.items():
        pat = r'\b(?:' + '|'.join(meta['keywords']) + r')\b'
        if re.search(pat, text):
            matched.append(asp)
    return "; ".join(matched) if matched else "Isu Umum Kebijakan"

df_v2['aspects'] = df_v2['text_clean'].apply(tag_aspects)

# Hitung statistik rekapitulasi V2
rows = []
for asp, meta in aspect_definitions.items():
    pat = r'\b(?:' + '|'.join(meta['keywords']) + r')\b'
    matched = df_v2[df_v2['text_clean'].str.contains(pat, regex=True, na=False)]
    
    total = len(matched)
    neg = len(matched[matched['sentiment'] == 'Negatif'])
    net = len(matched[matched['sentiment'] == 'Netral'])
    pos = len(matched[matched['sentiment'] == 'Positif'])
    pct_total = (total / len(df_v2)) * 100
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

# Simpan CSV V2
csv_v2 = os.path.join(OUTPUT_DIR, "aspek_kemiskinan_analisis_v2.csv")
res_df.sort_values(by='Total Entri', ascending=False).to_csv(csv_v2, index=False)
print(f"[+] Tabel CSV V2 berhasil disimpan ke: {csv_v2}")

# Simpan Dataset V2
dataset_v2_path = os.path.join(OUTPUT_DIR, "dataset_master_3000_v2.csv")
df_v2.to_csv(dataset_v2_path, index=False)

# Buat Stacked Bar Chart V2
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

y_pos = np.arange(len(res_df))
aspect_labels = res_df['Aspek Kemiskinan'].tolist()
neg_vals = res_df['Negatif (Protes)'].values
net_vals = res_df['Netral (Inkuiri)'].values
pos_vals = res_df['Positif (Syukur)'].values
total_vals = res_df['Total Entri'].values

c_neg = '#EF4444'  # Merah (Negatif)
c_net = '#3B82F6'  # Biru (Netral)
c_pos = '#10B981'  # Hijau (Positif)

p1 = ax.barh(y_pos, neg_vals, color=c_neg, edgecolor='#B91C1C', label='Sentimen Negatif (Keluhan/Protes)', height=0.55, alpha=0.92)
p2 = ax.barh(y_pos, net_vals, left=neg_vals, color=c_net, edgecolor='#1D4ED8', label='Sentimen Netral (Inkuiri/Informasi)', height=0.55, alpha=0.92)
p3 = ax.barh(y_pos, pos_vals, left=neg_vals + net_vals, color=c_pos, edgecolor='#047857', label='Sentimen Positif (Apresiasi/Syukur)', height=0.55, alpha=0.92)

for i in range(len(res_df)):
    if neg_vals[i] > 25:
        ax.text(neg_vals[i]/2, y_pos[i], f"{neg_vals[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    if net_vals[i] > 35:
        ax.text(neg_vals[i] + net_vals[i]/2, y_pos[i], f"{net_vals[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)
    if pos_vals[i] > 18:
        ax.text(neg_vals[i] + net_vals[i] + pos_vals[i]/2, y_pos[i], f"{pos_vals[i]}", ha='center', va='center', color='white', fontweight='bold', fontsize=9.5)

    total = total_vals[i]
    pct = (total / len(df_v2)) * 100
    ax.text(total + 15, y_pos[i], f"Total: {total} ({pct:.1f}%)", ha='left', va='center', color='#1E293B', fontweight='bold', fontsize=9.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(aspect_labels, fontsize=10.5, fontweight='bold')
ax.set_xlabel('Frekuensi Entri Opini Warganet (N)', fontsize=11, fontweight='bold')
ax.set_title('Distribusi Sentimen per Aspek Pengelompokan Kemiskinan pada Kebijakan Desil (V2)\n(Penyempurnaan Keluhan Implisit & Leksikon Negatif N = 2.994)',
             fontsize=12.5, fontweight='bold', pad=15)
ax.set_xlim(0, 880)
ax.legend(frameon=True, facecolor='white', edgecolor='#CBD5E1', loc='lower right', fontsize=10)

img_v2_path = os.path.join(IMAGES_DIR, "gambar_aspek_kemiskinan_stacked_v2.png")
plt.savefig(img_v2_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"[+] Grafik Stacked Bar Chart V2 berhasil disimpan ke: {img_v2_path}")
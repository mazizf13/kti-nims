"""
Script: build_dataset_master.py
Deskripsi: Mengharmonisasi dan menyatukan data 3 platform (Twitter, Threads, Instagram)
           masing-masing tepat 1.000 data menjadi satu dataset master terstandarisasi (N = 3.000).
"""

import os
import re
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_DIR = os.path.join(BASE_DIR, "01_Data_Hasil")
OUT_DIR = os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "01_Dataset")
os.makedirs(OUT_DIR, exist_ok=True)

CSV_OUT = os.path.join(OUT_DIR, "dataset_master_3000_raw.csv")
EXCEL_OUT = os.path.join(OUT_DIR, "dataset_master_3000_raw.xlsx")

# 1. Load Twitter
print("[*] Memuat data Twitter...")
df_tw = pd.read_csv(os.path.join(RAW_DIR, "data_twitter_desil_1000_raw.csv"))
df_tw = df_tw.dropna(subset=["Teks Tweet"]).drop_duplicates(subset=["Teks Tweet"])
df_tw = df_tw.head(1000).copy()
df_tw_std = pd.DataFrame({
    "id": [f"TW_{i+1:04d}" for i in range(len(df_tw))],
    "platform": "Twitter",
    "author": df_tw["Username Penulis"].fillna("anonim"),
    "text_raw": df_tw["Teks Tweet"].astype(str).str.strip(),
    "timestamp": df_tw["Waktu Posting (WIB)"].fillna("-"),
    "url": df_tw["URL Tweet"].fillna("-")
})
print(f"    -> Twitter terstandarisasi: {len(df_tw_std)} data")

# 2. Load Threads
print("[*] Memuat data Threads...")
df_th = pd.read_csv(os.path.join(RAW_DIR, "data_threads_desil_1000_raw.csv"))
df_th = df_th.dropna(subset=["Teks Postingan Threads"]).drop_duplicates(subset=["Teks Postingan Threads"])
df_th = df_th.head(1000).copy()
df_th_std = pd.DataFrame({
    "id": [f"TH_{i+1:04d}" for i in range(len(df_th))],
    "platform": "Threads",
    "author": df_th["Akun Penulis"].fillna("anonim"),
    "text_raw": df_th["Teks Postingan Threads"].astype(str).str.strip(),
    "timestamp": df_th["Waktu Posting (WIB)"].fillna("-"),
    "url": df_th["URL Postingan"].fillna("-")
})
print(f"    -> Threads terstandarisasi: {len(df_th_std)} data")

# 3. Load Instagram
print("[*] Memuat data Instagram...")
df_ig = pd.read_csv(os.path.join(RAW_DIR, "data_instagram_desil_raw.csv"))
df_ig = df_ig.dropna(subset=["Teks_Komentar"]).drop_duplicates(subset=["Teks_Komentar"])
# Filter komentar yang memiliki panjang teks memadai (minimal 15 karakter)
df_ig = df_ig[df_ig["Teks_Komentar"].str.len() >= 15]
df_ig = df_ig.head(1000).copy()
df_ig_std = pd.DataFrame({
    "id": [f"IG_{i+1:04d}" for i in range(len(df_ig))],
    "platform": "Instagram",
    "author": df_ig["Akun_Komentator"].fillna("anonim"),
    "text_raw": df_ig["Teks_Komentar"].astype(str).str.strip(),
    "timestamp": df_ig["Waktu_Komentar"].fillna("-"),
    "url": df_ig["Post_URL"].fillna("-")
})
print(f"    -> Instagram terstandarisasi: {len(df_ig_std)} data")

# 4. Gabungkan Ketiga Platform
df_master = pd.concat([df_tw_std, df_th_std, df_ig_std], ignore_index=True)
print(f"\n[+] Total Gabungan Master: {len(df_master)} data (Masing-masing tepat 1.000 data)")

# Simpan ke CSV
df_master.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
print(f"[+] File CSV berhasil disimpan di: {CSV_OUT}")

# Simpan ke Excel Rapi
print(f"[*] Membuat file Excel terformat rapi: {EXCEL_OUT}...")
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Master_Dataset_3000"

headers = ["No", "ID", "Platform", "Akun Penulis", "Teks Mentah (Raw Text)", "Waktu Unggahan", "URL"]
ws.append(headers)

header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
thin_border = Border(
    left=Side(style='thin', color='D9D9D9'),
    right=Side(style='thin', color='D9D9D9'),
    top=Side(style='thin', color='D9D9D9'),
    bottom=Side(style='thin', color='D9D9D9')
)

for col_num in range(1, len(headers) + 1):
    cell = ws.cell(row=1, column=col_num)
    cell.fill = header_fill
    cell.font = header_font
    cell.alignment = Alignment(horizontal="center", vertical="center")

for idx, row in df_master.iterrows():
    row_vals = [
        idx + 1,
        row["id"],
        row["platform"],
        row["author"],
        row["text_raw"],
        row["timestamp"],
        row["url"]
    ]
    ws.append(row_vals)
    curr_row = ws.max_row
    
    # Warna stripe beda platform agar mudah dibaca
    # Twitter: F8FAFC, Threads: F1F5F9, Instagram: EFF6FF
    plat_color = "F8FAFC" if row["platform"] == "Twitter" else ("F1F5F9" if row["platform"] == "Threads" else "EFF6FF")
    row_fill = PatternFill(start_color=plat_color, fill_type="solid")
    
    for c_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=curr_row, column=c_idx)
        cell.fill = row_fill
        cell.font = Font(name="Calibri", size=10)
        cell.border = thin_border
        if c_idx in [1, 2, 3, 6]:
            cell.alignment = Alignment(horizontal="center", vertical="center")
        elif c_idx == 4:
            cell.alignment = Alignment(horizontal="left", vertical="center")
        elif c_idx == 5:
            cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        else:
            cell.alignment = Alignment(horizontal="left", vertical="center")

col_widths = {1: 6, 2: 12, 3: 14, 4: 20, 5: 65, 6: 22, 7: 35}
for c_idx, w in col_widths.items():
    ws.column_dimensions[get_column_letter(c_idx)].width = w

ws.freeze_panes = "A2"
wb.save(EXCEL_OUT)
print(f"[+] File Excel Master berhasil disimpan: {EXCEL_OUT}")

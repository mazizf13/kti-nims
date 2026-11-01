"""
Script: generate_resume_v2_docx.py
Deskripsi: Menghasilkan Resume_Eksperimen_Sentimen_Desil.docx Versi 2 (V2)
           yang memuat perbaikan 35 keluhan implisit & Stacked Bar Chart V2.
"""

import os
import shutil
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
IMG_DIR_1 = os.path.join(BASE_DIR, "08_Analisis_Aspek_Kemiskinan_V2", "images")
IMG_DIR_2 = os.path.join(BASE_DIR, "assets")
IMG_DIR_3 = os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "07_Artikel_Ilmiah", "images")

doc = docx.Document()

for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def style_table(table, col_widths, align_left_col1=True):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    header_tr = table.rows[0]._tr.get_or_add_trPr()
    header_tr.append(parse_xml(r'<w:tblHeader %s/>' % nsdecls('w')))
    
    for c_idx, cell in enumerate(table.rows[0].cells):
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(r'<w:shd %s w:fill="1E3A8A"/>' % nsdecls('w')))
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        if c_idx < len(col_widths):
            cell.width = col_widths[c_idx]
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
                run.font.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)

    for r_idx, row in enumerate(table.rows[1:], start=1):
        shd_color = "F1F5F9" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, cell in enumerate(row.cells):
            tcPr = cell._tc.get_or_add_tcPr()
            tcPr.append(parse_xml(r'<w:shd %s w:fill="%s"/>' % (nsdecls('w'), shd_color)))
            set_cell_margins(cell, top=100, bottom=100, left=140, right=140)
            if c_idx < len(col_widths):
                cell.width = col_widths[c_idx]
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if (c_idx <= 1 and align_left_col1) else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9.0)
                    run.font.color.rgb = RGBColor(30, 41, 59)

# 1. HEADER
p_top = doc.add_paragraph()
p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_top = p_top.add_run("RESUME EKSEKUTIF HASIL RISET & EKSPERIMEN (VERSI 2)")
run_top.font.name = "Times New Roman"
run_top.font.size = Pt(16)
run_top.font.bold = True
run_top.font.color.rgb = RGBColor(30, 58, 138)

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_after = Pt(14)
run_sub = p_sub.add_run("Klasifikasi Sentimen Publik terhadap Kebijakan Desil Ekonomi dan DTSEN pada Media Sosial Multi-Platform (Twitter/X, Threads, Instagram) [Penyempurnaan Keluhan Implisit]")
run_sub.font.name = "Times New Roman"
run_sub.font.size = Pt(10.5)
run_sub.font.italic = True
run_sub.font.color.rgb = RGBColor(71, 85, 105)

# Profil Peneliti
tbl_info = doc.add_table(rows=3, cols=2)
tbl_info.alignment = WD_TABLE_ALIGNMENT.CENTER
info_data = [
    ("Peneliti / Penulis Utama", "Nimas Ayu Florentina"),
    ("Afiliasi / Instansi", "Badan Pusat Statistik (BPS) Kabupaten Mojokerto"),
    ("Korpus Empiris Riil", "N = 2.994 Data Bersih (Twitter 999, Threads 995, Instagram 1.000)")
]
for r_idx, (k, v) in enumerate(info_data):
    tbl_info.rows[r_idx].cells[0].text = k
    tbl_info.rows[r_idx].cells[1].text = v
    for c_idx in range(2):
        cell = tbl_info.rows[r_idx].cells[c_idx]
        set_cell_margins(cell, top=80, bottom=80, left=120, right=120)
        tcPr = cell._tc.get_or_add_tcPr()
        tcPr.append(parse_xml(r'<w:shd %s w:fill="F8FAFC"/>' % nsdecls('w')))
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.name = "Times New Roman"
                run.font.size = Pt(9.5)
                if c_idx == 0:
                    run.font.bold = True
                    run.font.color.rgb = RGBColor(30, 58, 138)

doc.add_paragraph().paragraph_format.space_after = Pt(8)

def add_heading(text, level=2):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(12) if level == 2 else Pt(11)
    run.font.bold = True
    run.font.color.rgb = RGBColor(30, 58, 138)

def add_paragraph(text, italic=False, bold=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(10.5)
    run.font.italic = italic
    run.font.bold = bold
    return p

def find_img(filename):
    for d in [IMG_DIR_1, IMG_DIR_2, IMG_DIR_3]:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None

def add_image_with_caption(filename, caption, width=Inches(6.0)):
    full_path = find_img(filename)
    if full_path:
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(3)
        run_img = p_img.add_run()
        run_img.add_picture(full_path, width=width)
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(10)
        run_cap = p_cap.add_run(caption)
        run_cap.font.name = "Times New Roman"
        run_cap.font.size = Pt(9.0)
        run_cap.font.italic = True
        run_cap.font.color.rgb = RGBColor(71, 85, 105)

# 1. Komparasi 6 Model
add_heading("1. Tabel Komparasi Kinerja Lengkap 6 Metode (Data Uji N = 458)")
add_paragraph("Seluruh model dievaluasi secara adil pada partisi data uji independen yang sama persis (70% Train, 15% Val, 15% Test):")

tbl_models = doc.add_table(rows=7, cols=7)
headers_6 = ["No", "Metode Klasifikasi", "Kategori Paradigma", "Akurasi", "Precision", "Recall", "Macro-F1 (Utama)"]
for c, h in enumerate(headers_6):
    tbl_models.rows[0].cells[c].text = h

data_6 = [
    ("1", "K-NN (k=5, Cosine)", "Shallow ML (TF-IDF)", "70,31%", "63,85%", "54,19%", "56,68%"),
    ("2", "Naïve Bayes (Complement)", "Shallow ML (TF-IDF)", "71,40%", "65,08%", "64,72%", "64,42%"),
    ("3", "Rocchio (Nearest Centroid)", "Shallow ML (TF-IDF)", "70,96%", "63,82%", "67,51%", "64,46%"),
    ("4", "SVM (Linear SVC)", "Shallow ML (TF-IDF)", "84,50%", "82,85%", "77,23%", "79,62%"),
    ("5", "BiLSTM (Bidirectional RNN)", "Deep Learning", "86,90%", "83,47%", "80,79%", "81,96%"),
    ("6", "IndoBERT (Transformer)", "Pretrained Transformer", "94,98%", "92,80%", "93,88%", "93,29%")
]
for r_idx, row in enumerate(data_6, start=1):
    for c_idx, val in enumerate(row):
        tbl_models.rows[r_idx].cells[c_idx].text = val
style_table(tbl_models, [Inches(0.4), Inches(1.8), Inches(1.5), Inches(0.8), Inches(0.8), Inches(0.8), Inches(1.1)])

add_paragraph("• IndoBERT Juara Bertahan: Unggul mutlak di seluruh metrik (Akurasi 94,98% dan Macro-F1 93,29%) berkat mekanisme Self-Attention yang tangguh terhadap kalimat sarkasme dan slang.")
add_paragraph("• SVM adalah Model Tradisional Terbaik: Mencatat Macro-F1 79,62% dan Akurasi 84,50%, mendekati performa BiLSTM (81,96%).")
add_paragraph("• K-NN, Naïve Bayes, dan Rocchio Tertinggal (56%–64%): Karena representasi TF-IDF yang renggang pada teks pendek serta ketidakmampuan mengenali negasi majemuk.")

add_image_with_caption("gambar_komparasi_6_model.png", "Gambar 1. Grafik Komparasi Kinerja Lengkap 6 Metode Klasifikasi Sentimen", width=Inches(6.2))

# 2. Confusion Matrix
add_heading("2. Analisis Visual Confusion Matrix (Hasil Riil Google Colab Pro)")
add_paragraph("Matriks kekeliruan memperlihatkan keandalan IndoBERT dalam mengenali kelas minoritas (Positif) dan mayoritas (Netral & Negatif) tanpa mengalami bias:")
add_paragraph("• IndoBERT: Berhasil memprediksi 92 dari 102 sentimen Negatif (90,2%), 295 dari 300 Netral (98,3%), dan 48 dari 56 Positif (85,7%).")
add_paragraph("• BiLSTM: Terdapat 16 keluhan Negatif yang keliru ditebak sebagai Netral karena kalimat warganet memuat ungkapan semi-formal.")

add_image_with_caption("confusion_matrix_komparasi.png", "Gambar 2. Matriks Kekeliruan Komparatif (Confusion Matrix) BiLSTM vs IndoBERT pada Data Uji", width=Inches(6.2))

# 3. Ketahanan Lintas Platform
add_heading("3. Ketahanan Generalisasi Lintas Platform (Cross-Platform Robustness)")
tbl_cross = doc.add_table(rows=4, cols=5)
headers_cross = ["Platform Uji", "Karakteristik Bahasa", "BiLSTM F1", "IndoBERT F1", "Keunggulan IndoBERT"]
for c, h in enumerate(headers_cross):
    tbl_cross.rows[0].cells[c].text = h
data_cross = [
    ("Twitter (X) [N=153]", "Teks pendek, singkatan gaul, sarkasme politik", "78,76%", "95,57%", "+16,81% (Paling Masif)"),
    ("Threads [N=152]", "Teks naratif panjang, curahan hati personal", "85,04%", "92,00%", "+6,96%"),
    ("Instagram [N=153]", "Komentar spontan emosional di bawah berita", "80,05%", "91,80%", "+11,75%")
]
for r_idx, row in enumerate(data_cross, start=1):
    for c_idx, val in enumerate(row):
        tbl_cross.rows[r_idx].cells[c_idx].text = val
style_table(tbl_cross, [Inches(1.4), Inches(2.5), Inches(0.9), Inches(1.0), Inches(1.4)])

add_paragraph("💡 Analisis Ilmiah: Keunggulan terbesar IndoBERT terjadi di Twitter (+16,81%). Tokenisasi sub-kata (WordPiece) mampu memecah singkatan baru warganet (misal: dtsen_gaje -> ['dts', '##en', 'gaje']) sehingga maknanya tetap tertangkap.")
add_image_with_caption("gambar4_cross_platform_performance.png", "Gambar 3. Macro F1-Score pada Masing-Masing Platform Media Sosial", width=Inches(5.8))

# 4. Aspek Kemiskinan V2 (Penyempurnaan Keluhan Implisit)
add_heading("4. Analisis Sentimen Berbasis Aspek Kemiskinan Versi 2 (V2 - Penyempurnaan)")
add_paragraph("Pada Versi 2 ini, dilakukan integrasi 35 pola leksikon keluhan implisit warganet (misal: 'gara-gara BPS desil naik jadi tidak dapat bantuan', 'rumah tidak punya kendaraan tapi desil 6', 'laporan ABS', 'pejabat desil 1', dll.) sehingga 97 data yang sebelumnya ambigu berhasil direklasifikasi secara akurat ke sentimen Negatif:")

tbl_asp_v2 = doc.add_table(rows=6, cols=6)
headers_asp_v2 = ["No", "Aspek Kemiskinan", "Total Entri V2", "Proporsi (%)", "Negatif (Protes)", "Rasio Negatif V2 (%)"]
for c, h in enumerate(headers_asp_v2):
    tbl_asp_v2.rows[0].cells[c].text = h

data_asp_v2 = [
    ("1", "Bansos & Ketahanan Pangan", "734", "24,52%", "208 (Naik +41)", "28,34%"),
    ("2", "Administrasi & Validasi DTSEN", "672", "22,44%", "189 (Naik +32)", "28,12%"),
    ("3", "Ketenagakerjaan & Pendapatan", "438", "14,63%", "152 (Naik +28)", "34,70% (Tertinggi)"),
    ("4", "Energi & Perumahan (Listrik 1300 VA)", "216", "7,21%", "60 (Naik +12)", "27,78%"),
    ("5", "Pendidikan (KIP Kuliah)", "172", "5,74%", "59 (Naik +12)", "34,30%")
]
for r_idx, row in enumerate(data_asp_v2, start=1):
    for c_idx, val in enumerate(row):
        tbl_asp_v2.rows[r_idx].cells[c_idx].text = val
style_table(tbl_asp_v2, [Inches(0.4), Inches(2.2), Inches(1.0), Inches(1.0), Inches(1.3), Inches(1.3)])

add_paragraph("💡 Temuan Penting Versi 2 (V2):")
add_paragraph("• Rasio Penolakan Tertinggi: Aspek Ketenagakerjaan (34,70%) dan Pendidikan KIP Kuliah (34,30%) mencatatkan persentase sentimen negatif terbesar karena langsung menyangkut hilangnya hak beasiswa kuliah anak miskin dan beban nafkah buruh serabutan.")
add_paragraph("• Volume Keluhan Terbesar: Masalah pencairan bansos pangan (208 keluhan) dan validasi data DTSEN (189 keluhan) membuktikan urgensi pembenahan sistem pendataan di tingkat RT/RW dan kelurahan.")

add_image_with_caption("gambar_aspek_kemiskinan_stacked_v2.png", "Gambar 4. Grafik Batang Bertumpuk (Stacked Bar Chart) Distribusi Sentimen per Aspek Kemiskinan V2 [Merah: Negatif, Biru: Netral, Hijau: Positif]", width=Inches(6.2))

# 5. Rekomendasi Format Artikel
add_heading("5. Rekomendasi Pilihan Format Artikel untuk Mbak Nimas")
add_paragraph("📌 Opsi A (Fokus Utama IndoBERT vs BiLSTM + Baseline Pendukung) - [SANGAT DIREKOMENDASIKAN]:", bold=True)
add_paragraph("• Judul: 'Analisis Komparatif Model IndoBERT dan BiLSTM dalam Klasifikasi Sentimen Kebijakan Desil pada Multi-Platform Media Sosial'")
add_paragraph("• Kelebihan: Judul ringkas, fokus riset tajam ke perdebatan Transformer vs RNN, namun di bab pembahasan tetap menyajikan tabel 6 model dan aspek kemiskinan V2.")

add_paragraph("📌 Opsi B (Format Benchmarking Skala Penuh 6 Model):", bold=True)
add_paragraph("• Judul: 'Benchmarking Model IndoBERT, BiLSTM, dan Algoritma Klasik (SVM, Naïve Bayes, Rocchio, K-NN) dalam Analisis Sentimen Kebijakan Desil pada Korpus Multi-Platform'")

# Simpan ke target lokasi
targets = [
    os.path.join(BASE_DIR, "Resume_Eksperimen_Sentimen_Desil.docx"),
    os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "Resume_Eksperimen_Sentimen_Desil.docx"),
    os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "07_Artikel_Ilmiah", "Resume_Eksperimen_Sentimen_Desil.docx"),
    os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "Resume_Eksekutif_Eksperimen_Sentimen_Desil_Mbak_Nimas.docx")
]

for t in targets:
    doc.save(t)
    print("Saved:", t)

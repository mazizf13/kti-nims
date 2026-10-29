"""
Script: create_resume_docx.py
Deskripsi: Membuat dokumen Microsoft Word (.docx) resmi dan elegan berupa
           Resume Eksekutif Hasil Riset & Eksperimen 6 Model untuk Mbak Nimas Ayu Florentina.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DOCX_1 = os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "Resume_Eksekutif_Eksperimen_Sentimen_Desil_Mbak_Nimas.docx")
OUTPUT_DOCX_2 = os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "07_Artikel_Ilmiah", "Resume_Eksekutif_Eksperimen_Sentimen_Desil_Mbak_Nimas.docx")
IMG_DIR = os.path.join(BASE_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "07_Artikel_Ilmiah", "images")

doc = docx.Document()

# Margin 1 inch
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
    
    # Header Row
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

    # Data Rows
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

# ------------------------------------------------------------------------------
# 1. HEADER / JUDUL DOKUMEN
# ------------------------------------------------------------------------------
p_top = doc.add_paragraph()
p_top.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_top = p_top.add_run("RESUME EKSEKUTIF HASIL RISET & EKSPERIMEN LENGKAP")
run_top.font.name = "Times New Roman"
run_top.font.size = Pt(16)
run_top.font.bold = True
run_top.font.color.rgb = RGBColor(30, 58, 138)

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
p_sub.paragraph_format.space_after = Pt(14)
run_sub = p_sub.add_run("Klasifikasi Sentimen Publik terhadap Kebijakan Desil Ekonomi dan DTSEN pada Media Sosial Multi-Platform (Twitter/X, Threads, Instagram)")
run_sub.font.name = "Times New Roman"
run_sub.font.size = Pt(11)
run_sub.font.italic = True
run_sub.font.color.rgb = RGBColor(71, 85, 105)

# Kotak Profil Peneliti
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

doc.add_paragraph().paragraph_format.space_after = Pt(10)

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

def add_image_with_caption(filename, caption, width=Inches(6.0)):
    full_path = os.path.join(IMG_DIR, filename)
    if os.path.exists(full_path):
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

# ------------------------------------------------------------------------------
# 2. KOMPARASI 6 MODEL
# ------------------------------------------------------------------------------
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

style_table(tbl_models, [Inches(0.4), Inches(1.8), Inches(1.5), Inches(0.8), Inches(0.8), Inches(0.8), Inches(1.1)], align_left_col1=True)

add_paragraph("Temuan Penting:")
add_paragraph("• IndoBERT Juara Bertahan: Unggul mutlak di seluruh metrik (Akurasi 94,98% dan Macro-F1 93,29%) berkat mekanisme Self-Attention yang tangguh terhadap kalimat sarkasme dan slang.")
add_paragraph("• SVM adalah Model Tradisional Terbaik: Mencatat Macro-F1 79,62% dan Akurasi 84,50%, mendekati performa BiLSTM (81,96%).")
add_paragraph("• K-NN, Naïve Bayes, dan Rocchio Tertinggal (56%–64%): Karena representasi TF-IDF yang renggang pada teks pendek serta ketidakmampuan mengenali negasi majemuk.")

add_image_with_caption("gambar_komparasi_6_model.png", "Gambar 1. Grafik Komparasi Kinerja Lengkap 6 Metode Klasifikasi Sentimen", width=Inches(6.2))

# ------------------------------------------------------------------------------
# 3. CONFUSION MATRIX
# ------------------------------------------------------------------------------
add_heading("2. Analisis Visual Confusion Matrix (Hasil Riil Google Colab Pro)")
add_paragraph("Matriks kekeliruan memperlihatkan keandalan IndoBERT dalam mengenali kelas minoritas (Positif) dan mayoritas (Netral & Negatif) tanpa mengalami bias:")
add_paragraph("• IndoBERT: Berhasil memprediksi 92 dari 102 sentimen Negatif (90,2%), 295 dari 300 Netral (98,3%), dan 48 dari 56 Positif (85,7%).")
add_paragraph("• BiLSTM: Terdapat 16 keluhan Negatif yang keliru ditebak sebagai Netral karena kalimat warganet memuat ungkapan semi-formal.")

add_image_with_caption("gambar5_confusion_matrix.png", "Gambar 2. Matriks Kekeliruan Komparatif (Confusion Matrix) BiLSTM vs IndoBERT pada Data Uji", width=Inches(6.2))

# ------------------------------------------------------------------------------
# 4. KETAHANAN LINTAS PLATFORM
# ------------------------------------------------------------------------------
add_heading("3. Ketahanan Generalisasi Lintas Platform (Cross-Platform Robustness)")
add_paragraph("Menilai seberapa tangguh model saat menghadapi karakteristik penulisan warganet yang berbeda di tiap media sosial:")

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

style_table(tbl_cross, [Inches(1.4), Inches(2.5), Inches(0.9), Inches(1.0), Inches(1.4)], align_left_col1=True)

add_paragraph("💡 Analisis Ilmiah: Keunggulan terbesar IndoBERT terjadi di Twitter (+16,81%). Tokenisasi sub-kata (WordPiece) mampu memecah singkatan baru warganet (misal: dtsen_gaje -> ['dts', '##en', 'gaje']) sehingga maknanya tetap tertangkap, sementara BiLSTM menganggapnya sebagai kata tidak dikenal (Out-of-Vocabulary).")

add_image_with_caption("gambar4_cross_platform_performance.png", "Gambar 3. Macro F1-Score pada Masing-Masing Platform Media Sosial", width=Inches(5.8))

# ------------------------------------------------------------------------------
# 5. DISTRIBUSI & TEMATIK
# ------------------------------------------------------------------------------
add_heading("4. Profil Opini Publik & Analisis Tematik Isu Keluhan")
add_paragraph("Dari 2.994 data warganet yang dianalisis:")
add_paragraph("• Sentimen Netral (65,23%): Mendominasi karena kebutuhan warganet untuk mengecek NIK dan cara verifikasi data DTSEN.")
add_paragraph("• Sentimen Negatif (22,18%): Keluhan masyarakat atas indikator desil yang dinilai keliru.")
add_paragraph("• Sentimen Positif (12,59%): Ungkapan rasa syukur atas pencairan bansos pangan dan beasiswa.")

add_image_with_caption("gambar2_distribusi_sentimen.png", "Gambar 4. Distribusi Proporsi Sentimen Publik Gabungan dan per Platform", width=Inches(5.8))

add_paragraph("Isu keluhan utama pada sentimen Negatif:")
add_paragraph("1. Anomali Daya Listrik PLN 1300 VA (29,4%) - Rumah kontrakan sederhana otomatis terlempar ke desil tinggi.")
add_paragraph("2. Ancaman Gugurnya Beasiswa KIP Kuliah (24,8%) - Mahasiswa miskin kehilangan bantuan pendidikan.")
add_paragraph("3. Ketimpangan & Exclusion Error (18,5%) - Tetangga mampu dapat desil rendah, warga miskin terdata desil tinggi.")

add_image_with_caption("gambar6_topik_keluhan_negatif.png", "Gambar 5. Distribusi Tematik Isu Keluhan Publik pada Sentimen Negatif", width=Inches(5.8))

# ------------------------------------------------------------------------------
# 5. ASPEK KEMISKINAN (ABSA STACKED BAR)
# ------------------------------------------------------------------------------
add_heading("5. Analisis Sentimen Berbasis Aspek Kemiskinan (ABSA)")
add_paragraph("Mengelompokkan opini publik ke dalam 5 Aspek Indikator Kemiskinan (Regsosek/BPS & DTSEN) untuk mengetahui aspek mana yang paling banyak dibicarakan dan menuai keluhan:")

tbl_asp = doc.add_table(rows=6, cols=6)
headers_asp = ["No", "Aspek Kemiskinan", "Total Entri", "Proporsi (%)", "Negatif (Protes)", "Positif (Syukur)"]
for c, h in enumerate(headers_asp):
    tbl_asp.rows[0].cells[c].text = h

data_asp = [
    ("1", "Bansos & Ketahanan Pangan", "734", "24,52%", "167", "204 (Tertinggi)"),
    ("2", "Administrasi & Validasi DTSEN", "637", "21,28%", "157", "147"),
    ("3", "Ketenagakerjaan & Pendapatan", "397", "13,26%", "124 (31,2% Rejection)", "27"),
    ("4", "Energi & Perumahan (Listrik 1300 VA)", "215", "7,18%", "48", "28"),
    ("5", "Pendidikan (KIP Kuliah)", "172", "5,74%", "47 (27,3% Rejection)", "24")
]
for r_idx, row in enumerate(data_asp, start=1):
    for c_idx, val in enumerate(row):
        tbl_asp.rows[r_idx].cells[c_idx].text = val

style_table(tbl_asp, [Inches(0.4), Inches(2.2), Inches(0.9), Inches(1.0), Inches(1.2), Inches(1.3)], align_left_col1=True)

add_paragraph("💡 Temuan Kebijakan:")
add_paragraph("• Volume Terbanyak: Bansos Pangan (24,52%) menyumbang percakapan dan apresiasi tertinggi (204 sentimen positif).")
add_paragraph("• Keluhan Terbanyak: Masalah validasi data DTSEN (157 keluhan negatif) akibat exclusion error pendataan RT/RW.")
add_paragraph("• Rasio Penolakan Tertinggi: Ketenagakerjaan (31,23% Negatif) dan KIP Kuliah (27,33% Negatif) karena menyangkut hak pendidikan anak miskin.")

add_image_with_caption("gambar_aspek_kemiskinan_stacked.png", "Gambar 6. Grafik Batang Bertumpuk (Stacked Bar Chart): Distribusi Sentimen per Aspek Kemiskinan [Merah: Negatif, Biru: Netral, Hijau: Positif]", width=Inches(6.2))

# ------------------------------------------------------------------------------
# 6. OPSI FORMAT ARTIKEL
# ------------------------------------------------------------------------------
add_heading("6. Rekomendasi Pilihan Format Artikel untuk Mbak Nimas")
add_paragraph("Mbak Nimas dapat memilih salah satu dari 2 pendekatan penyajian naskah publikasi:")

add_paragraph("📌 Opsi A (Fokus Utama IndoBERT vs BiLSTM + Baseline Pendukung) - [SANGAT DIREKOMENDASIKAN]:", bold=True)
add_paragraph("• Judul: 'Analisis Komparatif Model IndoBERT dan BiLSTM dalam Klasifikasi Sentimen Kebijakan Desil pada Multi-Platform Media Sosial'")
add_paragraph("• Kelebihan: Judul ringkas, fokus riset tajam ke perdebatan Transformer vs Recurrent Neural Network, namun di bab pembahasan tetap menyajikan tabel 6 model sebagai bukti bahwa IndoBERT juga mengalahkan algoritma klasik (SVM, Naïve Bayes, Rocchio, KNN).")

add_paragraph("📌 Opsi B (Format Benchmarking Skala Penuh 6 Model):", bold=True)
add_paragraph("• Judul: 'Benchmarking Model IndoBERT, BiLSTM, dan Algoritma Klasik (SVM, Naïve Bayes, Rocchio, K-NN) dalam Analisis Sentimen Kebijakan Desil pada Korpus Multi-Platform'")
add_paragraph("• Kelebihan: Menegaskan bahwa riset ini adalah tolok ukur komprehensif dari 3 generasi teknologi NLP (Klasik, Deep Learning, Transformer).")

# ------------------------------------------------------------------------------
# 7. TAUTAN BERKAS & DEMO WEB
# ------------------------------------------------------------------------------
add_heading("7. Akses Berkas Naskah & Aplikasi Demo Web")
add_paragraph("Seluruh file naskah dan aplikasi telah siap digunakan:")
add_paragraph("1. Naskah Word Artikel Ilmiah: 07_Artikel_Ilmiah/Artikel_Ilmiah_IndoBERT_vs_BiLSTM.docx")
add_paragraph("2. Naskah Markdown: 07_Artikel_Ilmiah/Artikel_Ilmiah_IndoBERT_vs_BiLSTM.md")
add_paragraph("3. Naskah Skripsi Asli (Lengkap Bab 1-5): 06_Naskah_Skripsi/Skripsi_IndoBERT_vs_BiLSTM.docx")
add_paragraph("4. Aplikasi Demo Web Streamlit Siap Deploy: Buka folder 05_Demo_Web_Streamlit/ lalu klik dua kali run_demo.bat untuk menjalankan antarmuka web interaktif.")

# Simpan ke dua lokasi
doc.save(OUTPUT_DOCX_1)
doc.save(OUTPUT_DOCX_2)
print(f"[+] Berhasil membuat resume eksekutif Word ke:\n    - {OUTPUT_DOCX_1}\n    - {OUTPUT_DOCX_2}")

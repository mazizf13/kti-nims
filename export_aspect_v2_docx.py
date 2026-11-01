"""
Script: export_aspect_v2_docx.py
Deskripsi: Mengekspor laporan analisis aspek kemiskinan V2 (Penyempurnaan Keluhan Implisit)
           ke format Word (.docx) lengkap dengan tabel komparasi sebelum vs sesudah dan gambar grafik baru.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "08_Analisis_Aspek_Kemiskinan_V2")
DOCX_PATH = os.path.join(OUTPUT_DIR, "Laporan_Analisis_Aspek_Kemiskinan_V2.docx")
IMG_PATH = os.path.join(OUTPUT_DIR, "images", "gambar_aspek_kemiskinan_stacked_v2.png")

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

def style_table(table, col_widths):
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
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx <= 1 else WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(9.0)
                    run.font.color.rgb = RGBColor(30, 41, 59)

# Judul
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = p.add_run("PENYEMPURNAAN ANALISIS ASPEK KEMISKINAN (V2)")
run.font.name = "Times New Roman"
run.font.size = Pt(15)
run.font.bold = True
run.font.color.rgb = RGBColor(30, 58, 138)

p_sub = doc.add_paragraph()
p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run_sub = p_sub.add_run("Integrasi 35 Pola Keluhan Implisit, Sindiran Birokrasi, dan Leksikon Negatif Warganet (N = 2.994)")
run_sub.font.name = "Times New Roman"
run_sub.font.size = Pt(10.5)
run_sub.font.italic = True
run_sub.font.color.rgb = RGBColor(71, 85, 105)

doc.add_paragraph()

# Paragraf Pembuka
p1 = doc.add_paragraph()
p1.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p1.paragraph_format.line_spacing = 1.15
r1 = p1.add_run("Analisis aspek kemiskinan versi awal (V1) disempurnakan menjadi Versi 2 (V2) setelah dilakukan audit terhadap komentar keluhan implisit. Sebelumnya, kalimat-kalimat seperti 'gara-gara BPS desil naik jadi tidak dapat bantuan', 'rumah tidak punya kendaraan tidak ada kerjaan tapi desil 6', atau 'laporan ABS semua' sempat tergolong Netral karena tidak memuat kata umpatan formal. Dengan mengintegrasikan 35 aturan keluhan negatif baru, sebanyak 97 data yang sebelumnya ambigu berhasil direklasifikasi secara presisi ke dalam sentimen Negatif.")
r1.font.name = "Times New Roman"
r1.font.size = Pt(10.5)

# Gambar Stacked Bar V2
if os.path.exists(IMG_PATH):
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_img = p_img.add_run()
    run_img.add_picture(IMG_PATH, width=Inches(6.2))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_cap = p_cap.add_run("Gambar 1. Grafik Batang Bertumpuk (Stacked Bar Chart) Distribusi Sentimen per Aspek Kemiskinan Versi 2 (V2)")
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(9.0)
    r_cap.font.italic = True
    r_cap.font.color.rgb = RGBColor(71, 85, 105)

# Tabel Komparasi V1 vs V2
p_t = doc.add_paragraph()
p_t.paragraph_format.space_before = Pt(12)
r_t = p_t.add_run("Tabel 1. Perbandingan Distribusi Sentimen Negatif Antara Versi 1 (Lama) vs Versi 2 (Penyempurnaan):")
r_t.font.name = "Times New Roman"
r_t.font.size = Pt(10.5)
r_t.font.bold = True

tbl = doc.add_table(rows=6, cols=6)
headers = ["Aspek Kemiskinan", "Total Entri V2", "Negatif V1 (Lama)", "Negatif V2 (Baru)", "Kenaikan Keluhan", "Rasio Negatif V2"]
for c, h in enumerate(headers):
    tbl.rows[0].cells[c].text = h

data_comp = [
    ("Bansos & Ketahanan Pangan", "734", "167", "208", "+41 (+24.6%)", "28.34%"),
    ("Administrasi & Validasi DTSEN", "672", "157", "189", "+32 (+20.4%)", "28.12%"),
    ("Ketenagakerjaan & Pendapatan", "438", "124", "152", "+28 (+22.6%)", "34.70%"),
    ("Energi & Perumahan (Listrik/Rumah)", "216", "48", "60", "+12 (+25.0%)", "27.78%"),
    ("Pendidikan (KIP Kuliah)", "172", "47", "59", "+12 (+25.5%)", "34.30%")
]

for r_idx, row in enumerate(data_comp, start=1):
    for c_idx, val in enumerate(row):
        tbl.rows[r_idx].cells[c_idx].text = val

style_table(tbl, [Inches(2.2), Inches(0.8), Inches(1.0), Inches(1.0), Inches(1.1), Inches(0.9)])

# Kesimpulan
doc.add_paragraph()
p2 = doc.add_paragraph()
p2.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p2.paragraph_format.line_spacing = 1.15
r2 = p2.add_run("Poin Penting Hasil Penyempurnaan V2:\n"
                "1. Rasio Keluhan Tertinggi: Aspek Ketenagakerjaan & Pendapatan (34,70%) dan Pendidikan KIP Kuliah (34,30%) menjadi ranah dengan resistensi paling tinggi karena menyangkut daya tahan ekonomi harian dan hak masa depan pendidikan anak miskin.\n"
                "2. Volume Keluhan Terbesar: Validasi data DTSEN (189 keluhan) dan penundaan pencairan bansos pangan (208 keluhan) membuktikan masyarakat paling aktif menyuarakan ketidaktepatan pendataan dan keterlambatan bansos.\n"
                "3. File ini disimpan mandiri di folder 08_Analisis_Aspek_Kemiskinan_V2 agar Anda dan Mbak Nimas bisa membandingkan secara leluasa sebelum memutuskan apakah ingin dipakai di naskah utama atau tetap menggunakan versi sebelumnya.")
r2.font.name = "Times New Roman"
r2.font.size = Pt(10.0)

doc.save(DOCX_PATH)
print(f"[+] Dokumen Word Laporan V2 berhasil dibuat ke: {DOCX_PATH}")

# Analisis Komparatif Model IndoBERT dan BiLSTM dalam Klasifikasi Sentimen Kebijakan Desil pada Multi-Platform Media Sosial

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Ready%20to%20Deploy-FF4B4B.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Architecture-Transformer%20vs%20RNN-purple.svg)]()
[![Dataset](https://img.shields.io/badge/Dataset-N%20%3D%202.994%20Clean-success.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

> **Peneliti / Penulis Utama:** Nimas Ayu Florentina  
> **Institusi / Afiliasi:** Badan Pusat Statistik (BPS) Kabupaten Mojokerto  
> **Email Korespondensi:** *nimasayuflorentina@gmail.com*  

---

## 📌 Ringkasan Proyek Penelitian

Repositori ini memuat implementasi komprehensif penelitian komparatif pemrosesan bahasa alami (*Natural Language Processing*/NLP) untuk mengukur dinamika opini publik terhadap kebijakan **Desil Kesejahteraan (Desil 1–10) dan pemutakhiran Data Tunggal Sasaran Perlindungan Sosial (DTSEN)** di Indonesia.

Penelitian ini membandingkan kinerja **6 algoritma klasifikasi sentimen** dari 3 paradigma teknologi yang berbeda:
1. **Contextual Transfer Learning:** **IndoBERT** (*Pretrained Bidirectional Transformer* - `indobenchmark/indobert-base-p1`).
2. **Deep Learning Sekuensial:** **BiLSTM** (*Bidirectional Long Short-Term Memory* + *Word Embeddings*).
3. **Machine Learning Konvensional (Baseline TF-IDF):** **Support Vector Machine (SVM)**, **Complement Naïve Bayes (CNB)**, **Rocchio Classifier (*Nearest Centroid*)**, dan **K-Nearest Neighbors (K-NN)**.

Di samping evaluasi kinerja model, penelitian ini menyajikan **Analisis Sentimen Berbasis Aspek Kemiskinan (*Aspect-Based Sentiment Analysis*/ABSA)** yang mengelompokkan keluhan warganet ke dalam indikator multidimensi kemiskinan (pangan, validasi data, ketenagakerjaan, daya listrik 1300 VA, dan beasiswa KIP Kuliah), serta purwarupa aplikasi web interaktif berbasis **Streamlit**.

---

## 📊 Hasil Evaluasi Empiris (Data Uji Independen $N = 458$)

Seluruh model diuji pada data uji independen yang sama persis (partisi terstratifikasi: 70% Latih, 15% Validasi, 15% Uji):

| No | Model Klasifikasi | Kategori Paradigma | Akurasi (%) | Macro-Precision (%) | Macro-Recall (%) | **Macro F1-Score (%) [Utama]** | Waktu Training |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | **K-NN** ($k=5$, Cosine) | Shallow ML (TF-IDF) | 70,31% | 63,85% | 54,19% | **56,68%** | ~0,001 detik |
| 2 | **Naïve Bayes** (Complement) | Shallow ML (TF-IDF) | 71,40% | 65,08% | 64,72% | **64,42%** | ~0,042 detik |
| 3 | **Rocchio** (*Nearest Centroid*) | Shallow ML (TF-IDF) | 70,96% | 63,82% | 67,51% | **64,46%** | ~0,240 detik |
| 4 | **SVM** (*Linear SVC*) | Shallow ML (TF-IDF) | 84,50% | 82,85% | 77,23% | **79,62%** | ~0,028 detik |
| 5 | **BiLSTM** (*Bidirectional RNN*) | Deep Learning Sekuensial | 86,90% | 83,47% | 80,79% | **81,96%** | ~40 detik (CPU) |
| 6 | **IndoBERT** (*Transformer*) | Contextual Transfer Learning | **94,98%** | **92,80%** | **93,88%** | **93,29%** | ~128 detik (GPU) |

### 🌐 Ketahanan Generalisasi Lintas Platform (*Cross-Platform Robustness*)

| Platform Media Sosial | Karakteristik Gaya Bahasa | BiLSTM Macro-F1 | IndoBERT Macro-F1 | Keunggulan IndoBERT |
|:---|:---|:---:|:---:|:---:|
| **Twitter (X)** ($N = 153$) | Teks pendek, singkatan alay masif, sarkasme politik | 78,76% | **95,57%** | **+16,81%** *(Keunggulan Terbesar)* |
| **Threads** ($N = 152$) | Teks naratif panjang, curahan hati personal | 85,04% | **92,00%** | **+6,96%** |
| **Instagram** ($N = 153$) | Komentar spontan emosional di bawah berita | 80,05% | **91,80%** | **+11,75%** |

---

## 🎯 Analisis Aspek Kemiskinan (ABSA)

Korpus opini warganet ($N = 2.994$) dipetakan ke dalam 5 indikator aspek kemiskinan:

| No | Aspek Kemiskinan | Total Entri ($N$) | Proporsi (%) | Negatif (Protes) | Netral (Inkuiri) | Positif (Syukur) | Rasio Negatif (%) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Bansos & Ketahanan Pangan** | **734** | 24,52% | 167 | 363 | **204** | 22,75% |
| 2 | **Administrasi & Validasi DTSEN** | **637** | 21,28% | **157** | 333 | 147 | 24,65% |
| 3 | **Ketenagakerjaan & Pendapatan** | **397** | 13,26% | 124 | 246 | 27 | **31,23%** |
| 4 | **Energi & Perumahan (Listrik 1300 VA)** | **215** | 7,18% | 48 | 139 | 28 | 22,33% |
| 5 | **Pendidikan (KIP Kuliah)** | **172** | 5,74% | 47 | 101 | 24 | **27,33%** |

* **Volume Terbesar:** Aspek *Bansos & Pangan* (24,5%) menyumbang apresiasi positif tertinggi atas pencairan bantuan beras.
* **Keluhan Terbesar:** Aspek *Administrasi DTSEN* (157 keluhan) akibat ketidakakuratan data pemeringkatan desil di tingkat daerah.
* **Rasio Penolakan Kritis:** *Ketenagakerjaan* (31,2% Negatif) dan *KIP Kuliah* (27,3% Negatif) karena menyangkut ancaman putus kuliah mahasiswa prasejahtera akibat anomali indikator daya listrik.

---

## 🚀 Panduan Menjalankan Aplikasi Web (Streamlit)

Aplikasi telah dipaketkan secara **mandiri (*self-contained*)** dengan ukuran sangat ringan (**3,78 MB**), siap dijalankan di komputer lokal maupun di-deploy ke cloud gratis.

### A. Menjalankan di Komputer Lokal (Windows)
1. Cukup klik dua kali file batch:
   ```cmd
   run_demo_web.bat
   ```
2. Atau jalankan melalui terminal:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Browser akan terbuka otomatis di alamat `http://localhost:8501`.

### B. Deploy ke Streamlit Community Cloud (Gratis & Online)
1. Push/unggah seluruh isi repositori ini ke **GitHub** Anda.
2. Masuk ke [share.streamlit.io](https://share.streamlit.io/) menggunakan akun GitHub.
3. Klik tombol **New App**, lalu isi:
   * **Repository:** Nama repositori GitHub Anda.
   * **Branch:** `main` (atau `master`).
   * **Main file path:** `app.py`.
4. Klik **Deploy!** Dalam 1–2 menit, web aplikasi interaktif Anda langsung tayang dan dapat diakses publik.

---

## 📁 Struktur Direktori Repositori

```text
├── app.py                             <- Aplikasi Web Streamlit Utama (5 Tab Navigasi)
├── requirements.txt                   <- Daftar dependensi Python untuk Streamlit Cloud
├── run_demo_web.bat                   <- Launcher 1-klik aplikasi web Streamlit
├── .gitignore                         <- Konfigurasi proteksi filter file besar (>100MB)
├── data/                              <- Dataset CSV preprocessed teranotasi aspek & CSV evaluasi
├── assets/                            <- Grafik visualisasi publikasi 300 DPI, stacked bar, & confusion matrix
├── saved_classical_models/            <- Bobot model .pkl terlatih (TF-IDF, SVM, NB, KNN, Rocchio)
│
├── 📂 00_Skripsi_IndoBERT_vs_BiLSTM/
│   ├── 📂 01_Dataset/                 <- Dataset master raw & preprocessed (Excel & CSV), train/val/test splits
│   ├── 📂 02_Preprocessing/           <- Script normalisasi teks & kamus_slang_indonesia.json (300+ kata)
│   ├── 📂 03_Notebooks_Colab/         <- Google Colab Pro Training Notebook (GPU)
│   ├── 📂 04_Scripts_Training_Lokal/  <- Skrip training lokal & evaluasi komparatif model
│   ├── 📂 05_Demo_Web_Streamlit/      <- Direktori salinan mandiri aplikasi demo web Streamlit
│   ├── 📂 06_Naskah_Skripsi/          <- Naskah lengkap Skripsi Bab 1–5 (.docx & .md)
│   ├── 📂 07_Artikel_Ilmiah/          <- Naskah Paper/Artikel Ilmiah Jurnal (.docx & .md) + Gambar 300 DPI
│   ├── Resume_Eksekutif_Eksperimen_Sentimen_Desil_Mbak_Nimas.docx <- Dokumen Word Resume Resmi
│   └── run_classical_models.bat       <- Launcher 1-klik training 4 model klasik lokal
│
└── README.md                          <- Dokumentasi proyek penelitian ini
```

---

## 📚 Sitasi / Rujukan Akademis

Jika Anda menggunakan dataset, kode program, atau metodologi dari penelitian ini, silakan sitasi naskah artikel berikut:

```bibtex
@article{florentina2026indobert,
  author    = {Florentina, Nimas Ayu},
  title     = {Analisis Komparatif Model IndoBERT dan BiLSTM dalam Klasifikasi Sentimen Kebijakan Desil pada Multi-Platform Media Sosial},
  journal   = {Badan Pusat Statistik Kabupaten Mojokerto},
  year      = {2026},
  pages     = {1--18}
}
```

---

*Hak Cipta © 2026 Nimas Ayu Florentina – Badan Pusat Statistik (BPS) Kabupaten Mojokerto.*

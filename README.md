# Analisis Sentimen Kebijakan Desil & DTSEN (Twitter & Threads)

Repositori penelitian ini memuat data empiris, skrip scraping, analisis sentimen berbasis leksikon, serta draf Karya Tulis Ilmiah (KTI) mengenai evaluasi persepsi masyarakat Indonesia terhadap kebijakan **Desil Kesejahteraan (Desil 1–10) & DTSEN**.

---

## 📁 Struktur Direktori Proyek

```text
threads/
│
├── 📂 01_Data_Hasil/           <- Seluruh file dataset hasil scraping & analisis
│   ├── data_hasil_analisis_sentimen_desil_2001.xlsx  <- Dataset gabungan lengkap berlabel sentimen (2.001 data)
│   ├── data_analisis_sentimen_desil_gabungan.csv     <- Format CSV analisis sentimen gabungan
│   ├── data_twitter_desil_1000_rapi.xlsx             <- Dataset Twitter/X rapi & terstruktur (1.001 tweet)
│   ├── data_twitter_desil_1000_raw.csv               <- Dataset Twitter mentah
│   ├── data_threads_desil_1000_rapi.xlsx             <- Dataset Threads rapi & terstruktur (1.000 postingan)
│   └── data_threads_desil_1000_raw.csv               <- Dataset Threads mentah
│
├── 📂 02_Naskah_KTI/           <- Naskah Karya Tulis Ilmiah / Artikel Lengkap
│   └── KTI_Analisis_Sentimen_Desil_DTSEN.md          <- Format IMRaD siap publikasi/jurnal
│
├── 📂 03_Scripts_Python/       <- Skrip otomasi Python (Scraper & Sentiment Pipeline)
│   ├── twitter_scraper.py                            <- Scraper Twitter/X (Playwright, Anti-Limit, SNA)
│   ├── threads_scraper.py                            <- Scraper Threads (Playwright, Anti-Spam India, SNA)
│   ├── sentiment_analysis.py                         <- Pipeline Lexicon-Based Sentiment Analysis
│   └── format_excel.py                               <- Generator styling Excel otomatis
│
├── 📂 04_Scripts_R/            <- Alternatif skrip pemrosesan dengan bahasa R
│   ├── threads_api_scraper.R                         <- Metode resmi Meta API (httr2)
│   └── threads_chromote_scraper.R                    <- Metode headless Chrome di R (chromote)
│
├── 📂 05_Sessions/             <- Folder penyimpanan sesi login browser (Persistent Auth)
│   ├── threads_session/                              <- Cookies & sesi Threads
│   └── twitter_session/                              <- Cookies & sesi Twitter
│
├── 📂 99_Arsip_Testing/        <- Skrip uji coba internal, network inspector & screenshot
│
└── README.md                   <- Dokumentasi struktur & panduan repositori
```

---

## 📊 Ringkasan Korpus Data ($N = 2.001$)

| Platform | Total Data | Sentimen Negatif | Sentimen Netral | Sentimen Positif | Akun Target Terbanyak (SNA) |
|---|---|---|---|---|---|
| **Twitter (X)** | **1.001** | 41,36% | 45,05% | 13,59% | `@bps_statistics`, `@prabowo`, `@KemensosRI` |
| **Threads** | **1.000** | 43,90% | 36,00% | 20,10% | `@kemensosri`, `@acyersam`, `@bapenda.jabar` |
| **GABUNGAN** | **2.001** | **42,63%** | **40,53%** | **16,84%** | `@bps_statistics` (45 interaksi) |

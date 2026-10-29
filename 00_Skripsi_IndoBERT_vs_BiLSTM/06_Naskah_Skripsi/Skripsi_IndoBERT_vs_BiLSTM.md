# Analisis Komparatif Model IndoBERT dan BiLSTM dalam Klasifikasi Sentimen Publik terhadap Kebijakan Desil pada Media Sosial Threads, Instagram, dan Twitter/X

**Penulis:** Florentina, Nimas Ayu  
**Institusi/Afiliasi:** BPS Kabupaten Mojokerto  
**Email:** nimasayuflorentina@gmail.com  

---

## ABSTRAK

Penerapan klasifikasi desil kesejahteraan (Desil 1–10) dan pemutakhiran Data Tunggal Sasaran Perlindungan Sosial (DTSEN) di Indonesia memicu dinamika diskursus publik yang masif di berbagai platform media sosial. Penelitian ini bertujuan untuk membandingkan performa dua arsitektur *deep learning* yang berbeda secara paradigmatis, yaitu **IndoBERT** (*pretrained contextual transformer*) dan **BiLSTM** (*bidirectional recurrent neural network* dengan *word embedding*), dalam mengklasifikasikan polaritas sentimen publik (Positif, Negatif, Netral) terhadap kebijakan desil ekonomi. Korpus empiris dihimpun secara seimbang dari tiga platform media sosial utama di Indonesia, yaitu Twitter/X ($N = 1.000$), Threads ($N = 1.000$), dan Instagram ($N = 1.000$), sehingga membentuk dataset terstandarisasi sebesar $N = 3.000$ entri. Tahap prapengolahan data mencakup pembersihan derau, *case folding*, dan normalisasi singkatan/slang warganet menggunakan kamus leksikon informal bahasa Indonesia. Dataset dibagi secara terstratifikasi (*stratified splitting*) dengan proporsi 70% data latih ($N = 2.092$), 15% data validasi ($N = 444$), dan 15% data uji ($N = 458$). Hasil eksperimen membuktikan bahwa **IndoBERT secara konsisten mengungguli BiLSTM di seluruh metrik evaluasi**, dengan meraih **Macro-F1 sebesar 93,29%** dan **Akurasi 94,98%**, dibandingkan BiLSTM yang mencatat **Macro-F1 81,96%** dan **Akurasi 86,90%** (selisih keunggulan Macro-F1 mencapai $+11,33\%$). Pada pengujian generalisasi lintas platform (*cross-platform evaluation*), IndoBERT menunjukkan keunggulan paling mencolok pada platform Twitter/X ($+16,81\%$ Macro-F1 vs BiLSTM, dengan F1 mencapai 95,57%) berkat mekanisme *bidirectional self-attention* yang sangat tangguh dalam mengekstraksi semantik kalimat pendek bersarkasme dan ber-slang tinggi. Secara tematik, polaritas negatif publik terkonsentrasi pada anomali variabel daya listrik 1300 VA dan terhambatnya beasiswa KIP Kuliah bagi mahasiswa prasejahtera. Sebagai luaran praktis, dikembangkan purwarupa aplikasi web interaktif berbasis Streamlit untuk mendemonstrasikan inferensi sentimen model secara waktu nyata.

**Kata Kunci:** *IndoBERT, BiLSTM, Analisis Sentimen, Desil Kesejahteraan, DTSEN, Media Sosial, Multi-Platform, Deep Learning.*

---

## BAB 1. PENDAHULUAN

### 1.1 Latar Belakang

Transformasi digital dalam tata kelola jaring pengaman sosial di Indonesia ditandai dengan integrasi basis data kemiskinan tunggal melalui Data Tunggal Sasaran Perlindungan Sosial (DTSEN) yang memadukan hasil Registrasi Sosial Ekonomi (Regsosek) dengan Data Terpadu Kesejahteraan Sosial (DTKS) (Kementerian PPN/Bappenas, 2023). Pemerintah melalui Kementerian Sosial (Kemensos) dan Badan Pusat Statistik (BPS) menetapkan pemeringkatan desil kesejahteraan—yang mengelompokkan rumah tangga dari Desil 1 (10% termiskin) hingga Desil 10 (10% terkaya)—sebagai instrumen mutlak kelayakan program perlindungan sosial seperti Program Keluarga Harapan (PKH), Bantuan Pangan Non-Tunai (BPNT), hingga bantuan pendidikan tinggi Kartu Indonesia Pintar Kuliah (KIP-K) (Badan Pusat Statistik, 2024; World Bank, 2022).

Kendati dirancang secara matematis untuk menekan angka galat inklusi (*inclusion error*) dan galat eksklusi (*exclusion error*), implementasi kebijakan pemeringkatan desil di tingkat masyarakat menuai gelombang resistensi dan diskursus kritis yang sangat dinamis (Prasetyo & Rahayu, 2023). Publik mengeluhkan adanya diskrepansi mencolok antara kondisi riil ekonomi keluarga dengan status desil yang tercatat di sistem digital. Fenomena anomali indikator fisik (misalnya rumah kontrakan keluarga tidak mampu yang terpasang daya listrik PLN 1300 VA) dilaporkan secara otomatis mendongkrak status rumah tangga ke Desil 6–10, yang berakibat fatal pada pembatalan sepihak beasiswa KIP Kuliah maupun pemutusan bansos pangan.

Media sosial telah berevolusi menjadi ruang deliberatif publik informal (*informal digital public sphere*) di mana masyarakat secara organik membagikan keluhan, mencari informasi administratif, dan menuntut transparansi pemerintah (Fuchs, 2021). Menariknya, diskursus kebijakan desil ini tersebar di tiga media sosial utama dengan karakteristik linguistik yang sangat beraneka ragam:
1. **Twitter (X)**: Ditandai oleh teks mikroblog yang ringkas, kaya singkatan gaul (*slang*), penggunaan sarkasme politik, dan *direct mention* ke akun otoritas negara (`@bps_statistics`, `@KemensosRI`).
2. **Threads**: Ditandai oleh teks naratif panjang (*storytelling* personal) berupa curahan hati mendalam mengenai kemiskinan faktual keluarga.
3. **Instagram**: Ditandai oleh komentar spontan dan emosional yang terkumpul di bawah unggahan visual/infografis edukasi dan berita resmi.

Dari sudut pandang Pemrosesan Bahasa Alami (*Natural Language Processing*/NLP), keberagaman register bahasa media sosial Indonesia menghadirkan tantangan komputasi yang tinggi. Metode klasifikasi konvensional seperti *Lexicon-based* maupun algoritma *Machine Learning* klasik (*Support Vector Machine*, *Naïve Bayes*) dengan representasi *Bag-of-Words* memiliki keterbatasan mendasar: tidak memiliki kemampuan generalisasi semantik (*semantic generalization*) dan gagal menangkap dependensi kontekstual jarak jauh dalam kalimat majemuk (Liu, 2020; Taboada et al., 2011).

Untuk mengatasi hambatan tersebut, paradigma *Deep Learning* menawarkan terobosan mutakhir melalui dua arsitektur yang memiliki filosofi representasi berbeda:
* **IndoBERT**: Model berbasis *Transformer* yang memanfaatkan mekanisme *Bidirectional Self-Attention* dan telah dilatih (*pretrained*) pada miliaran kata korpus bahasa Indonesia (Koto et al., 2020; Wilie et al., 2020). IndoBERT mampu menghasilkan representasi kata yang terkontekstualisasi (*contextualized embeddings*), sehingga kata yang sama dapat memiliki makna representasi berbeda tergantung susunan kalimat di sekitarnya.
* **BiLSTM (Bidirectional Long Short-Term Memory)**: Model jaringan saraf tiruan berulang (*Recurrent Neural Network*) yang memproses sekuens teks secara dua arah (maju dan mundur) menggunakan unit gerbang (*gates*) memori (Hochreiter & Schmidhuber, 1997; Schuster & Paliwal, 1997). BiLSTM dikenal memiliki efisiensi komputasi yang lebih hemat dibanding Transformer dan terbukti andal dalam memodelkan urutan temporal teks (Fadillah & Wibowo, 2023).

Hingga saat ini, belum ada penelitian komparatif yang menguji secara komprehensif keandalan IndoBERT melawan BiLSTM pada korpus multi-platform kebijakan desil ekonomi di Indonesia. Oleh karena itu, penelitian ini dilakukan untuk mengisi kesenjangan metodologis tersebut, mengukur performa klasifikasi sentimen kedua model, menganalisis ketahanan generalisasi lintas platform (*cross-platform robustness*), serta membangun purwarupa aplikasi web interaktif sebagai luaran aplikatif.

### 1.2 Rumusan Masalah

Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian ini adalah:
1. Bagaimana karakteristik distribusi opini dan sentimen publik terhadap kebijakan desil ekonomi pada korpus seimbang media sosial Twitter (X), Threads, dan Instagram?
2. Bagaimana perbandingan performa model IndoBERT dan BiLSTM dalam mengklasifikasikan sentimen publik berdasarkan metrik *Macro-F1*, *Accuracy*, *Precision*, dan *Recall*?
3. Bagaimana kemampuan generalisasi lintas platform (*cross-platform generalization*) dari model IndoBERT dan BiLSTM saat diuji pada masing-masing karakteristik platform media sosial?

### 1.3 Tujuan Penelitian

Penelitian ini bertujuan untuk:
1. Mengintegrasikan dan menganalisis korpus multi-platform sebesar $N = 3.000$ data terstratifikasi dari Twitter, Threads, dan Instagram terkait kebijakan desil dan DTSEN.
2. Membangun, melatih, dan membandingkan performa arsitektur *deep learning* IndoBERT (*Fine-Tuned Transformer*) dan BiLSTM (*Word Embedding*) pada data uji yang sama.
3. Menguji ketangguhan model dalam menangani variasi linguistik lintas platform media sosial.
4. Mengembangkan aplikasi demo interaktif berbasis Streamlit untuk pengujian sentimen publik secara waktu nyata (*real-time*).

### 1.4 Manfaat Penelitian

* **Manfaat Teoretis:** Memberikan kontribusi empiris dalam bidang *computational linguistics* dan NLP bahasa Indonesia mengenai efektivitas komparatif arsitektur *Attention-based* (Transformer) versus *Recurrent-based* (RNN) pada domain kebijakan publik sosial-ekonomi.
* **Manfaat Praktis:** Menyediakan instrumen evaluasi kebijakan (*policy feedback tool*) bagi Badan Pusat Statistik (BPS) dan Kementerian Sosial dalam memantau respons dan persepsi masyarakat secara objektif dan terukur.

---

## BAB 2. TINJAUAN PUSTAKA

### 2.1 Kebijakan Desil Ekonomi, DTSEN, dan Perlindungan Sosial

Konsep desil dalam statistik sosial membagi populasi rumah tangga berurutan ke dalam sepuluh fraksi persentil yang sama besar (masing-masing 10%) berdasarkan parameter pengeluaran per kapita (Badan Pusat Statistik, 2024):
* **Desil 1–4**: Mewakili 40% populasi terbawah yang dikategorikan sebagai miskin ekstrem, miskin, dan rentan miskin. Kelompok ini menjadi sasaran prioritas penerima jaring pengaman sosial nasional.
* **Desil 5**: Kelompok transisi (*lower-middle class*) yang sangat rentan jatuh miskin apabila terjadi guncangan ekonomi (*economic shock*).
* **Desil 6–10**: Kelompok menengah ke atas hingga kelompok mapan yang secara normatif dieksklusi dari bantuan sosial reguler.

Registrasi Sosial Ekonomi (Regsosek) yang diintegrasikan ke dalam DTSEN berfungsi sebagai *single source of truth* untuk menekan galat administratif penargetan bantuan (Coady et al., 2004; Kementerian PPN/Bappenas, 2023). Namun, dalam persepsi warganet, indikator penentu desil dinilai kerap bias karena menggunakan proksi fisik (daya listrik, kepemilikan aset keluarga besar) yang tidak mencerminkan daya beli riil individu pemohon (Prasetyo & Rahayu, 2023).

### 2.2 Karakteristik Linguistik Media Sosial Multi-Platform

Teks media sosial memiliki tantangan tersendiri yang membedakannya dari teks formal jurnalistik:
* **Twitter (X)**: Mengharuskan pengguna memadatkan pesan dalam batas karakter yang ketat, memicu munculnya akronim ekstrem (*"yg"*, *"bgt"*, *"tdk"*), ungkapan sarkastik, dan pencantuman *mention* akun pejabat publik (Yuliana & Hartanto, 2022).
* **Threads**: Ekosistem berbasis teks yang memfasilitasi penulisan naratif panjang tanpa batasan karakter mikro, sehingga warganet cenderung mencurahkan kisah faktual kehidupan keluarganya secara emosional dan terperinci.
* **Instagram**: Komentar berada di kolom bawah konten visual (foto/infografis). Karakteristik komentarnya sangat reaktif, singkat, dan terfokus pada poin-poin infografis yang diunggah akun media atau kementerian.

### 2.3 Arsitektur BiLSTM (Bidirectional Long Short-Term Memory)

BiLSTM merupakan pengembangan dari arsitektur LSTM standar yang diperkenalkan oleh Schuster & Paliwal (1997). Jika LSTM konvensional hanya membaca urutan kata dari kiri ke kanan ($t = 1 \rightarrow T$), BiLSTM menggabungkan dua lapisan sel tersembunyi yang bergerak secara simultan: satu lapisan memproses urutan maju ($\vec{h}_t$) dan lapisan lainnya memproses urutan mundur ($\overleftarrow{h}_t$).

Keluaran representasi tersembunyi pada setiap langkah waktu $t$ merupakan penggabungan (*concatenation*) dari kedua arah:

$$h_t = [\vec{h}_t \, ; \, \overleftarrow{h}_t]$$

Melalui struktur ini, representasi kata pada indeks tertentu mampu menangkap konteks historis kata-kata sebelumnya sekaligus konteks kata-kata setelahnya, sehingga menghasilkan representasi sekuensial yang kuat sebelum diteruskan ke lapisan *Fully-Connected Layer* dan *Softmax* untuk klasifikasi sentimen (Fadillah & Wibowo, 2023; Hochreiter & Schmidhuber, 1997).

### 2.4 Arsitektur IndoBERT (Transformer Pretrained)

IndoBERT adalah model bahasa berbasis arsitektur *Transformer* (Vaswani et al., 2017) yang dilatih khusus pada korpus teks bahasa Indonesia skala besar (*Indo4B*) yang mencakup berita, Wikipedia, dan media sosial (Koto et al., 2020; Wilie et al., 2020). Berbeda dari model sekuensial RNN/LSTM yang memproses teks secara bertahap kata demi kata, IndoBERT memproses seluruh kata dalam kalimat secara paralel menggunakan mekanisme *Scaled Dot-Product Attention*:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Model `indobenchmark/indobert-base-p1` tersusun atas 12 lapisan *Transformer Encoder*, 768 *hidden size*, dan 12 *attention heads* dengan total sekitar 110 juta parameter. Dalam klasifikasi sentimen, token khusus `[CLS]` (*Classification token*) yang berada di awal urutan input diekstraksi sebagai representasi agregat seluruh kalimat, lalu dihubungkan ke lapisan linear penentu probabilitas kelas sentimen melalui proses *fine-tuning* terarah (Liu, 2020).

---

## BAB 3. METODOLOGI PENELITIAN

### 3.1 Kerangka Kerja Penelitian

Penelitian ini dirancang secara sistematis melalui enam tahapan berurutan: (1) Penarikan dan Kurasi Korpus Seimbang Multi-Platform ($N = 3.000$), (2) Pra-pengolahan Teks & Normalisasi Slang, (3) Pelabelan Sentimen *Ground Truth*, (4) Pembagian Dataset Terstratifikasi (*Stratified Splitting* 70:15:15), (5) Pelatihan dan Penyetelan Model IndoBERT vs BiLSTM, serta (6) Evaluasi Komparatif Komprehensif dan Implementasi Purwarupa Web.

### 3.2 Sumber Data dan Desain Pengambilan Sampel

Korpus empiris dikumpulkan secara proporsional dengan prinsip *platform parity* (keseimbangan platform) guna menjamin keadilan statistik:
* **Twitter (X)**: $N = 1.000$ tweet unik.
* **Threads**: $N = 1.000$ unggahan unik.
* **Instagram**: $N = 1.000$ komentar publik terverifikasi.
* **Total Korpus Master**: $N = 3.000$ data teks.

Kata kunci pencarian difokuskan pada leksikon kebijakan perlindungan sosial: `"desil"`, `"dtsen"`, `"desil bansos"`, `"cek desil"`, dan `"desil kip kuliah"`.

### 3.3 Pra-pengolahan Data (*Text Preprocessing Pipeline*)

Tahapan prapengolahan data teks meliputi:
1. **Cleaning:** Menghilangkan tautan URL web, mention pengguna (`@username`), simbol tanda pagar (`#hashtag`), tanda baca khusus, angka redundan, dan spasi ganda.
2. **Case Folding:** Menyeragamkan seluruh karakter teks menjadi huruf kecil (*lowercase*).
3. **Slang & Abbreviation Normalization:** Memetakan singkatan informal warganet ke bentuk baku bahasa Indonesia menggunakan kamus slang (contoh: *"yg"* $\rightarrow$ *"yang"*, *"gak/ga/ngga"* $\rightarrow$ *"tidak"*, *"bgt"* $\rightarrow$ *"sangat"*).
4. **Catatan Teknis NLP:** Pada IndoBERT, teks **tidak dilakukan stemming** (*affix removal*) karena algoritma *WordPiece Tokenizer* pada BERT memerlukan imbuhan bahasa Indonesia untuk mengenali relasi gramatikal kontekstual secara presisi.

### 3.4 Pembagian Dataset Terstratifikasi (*Stratified Data Splitting*)

Dataset sebesar $N = 2.994$ data bersih dibagi ke dalam tiga subset terpisah secara terstratifikasi berdasarkan kombinasi platform dan label sentimen guna memastikan rasio distribusi kelas yang identik di setiap subset:
* **Data Latih (*Training Set* - 70%)**: $N = 2.092$ data, digunakan untuk melatih bobot model.
* **Data Validasi (*Validation Set* - 15%)**: $N = 444$ data, digunakan untuk *hyperparameter tuning* dan *early stopping*.
* **Data Uji (*Testing Set* - 15%)**: $N = 458$ data, diisolasi murni untuk pengujian akhir performa komparatif.

### 3.5 Pengaturan Hyperparameter Model

* **IndoBERT (`indobenchmark/indobert-base-p1`):**
  * Optimizer: AdamW (Learning Rate: $2 \times 10^{-5}$, Weight Decay: $0.01$).
  * Batch Size: 16, Maximum Sequence Length: 128 tokens.
  * Epoch: 4 epoch dengan *linear warmup scheduler*.
  * Loss Function: *Weighted Cross-Entropy Loss* untuk mengantisipasi ketidakseimbangan kelas (*class imbalance*).
* **BiLSTM:**
  * Embedding Dimension: 128, Hidden Units: 128 per arah (total 256).
  * Number of Layers: 2 bidirectional layers, Dropout Rate: 0.4.
  * Optimizer: Adam (Learning Rate: $1 \times 10^{-3}$, Batch Size: 32).
  * Epoch: 15 epoch dengan pemantauan *Early Stopping* pada data validasi.

### 3.6 Metrik Evaluasi

Mengingat distribusi kelas sentimen di media sosial bersifat tidak seimbang (*imbalanced*), **Macro-Averaged F1-Score (Macro-F1)** ditetapkan sebagai **metrik evaluasi utama**, didukung oleh *Accuracy*, *Macro-Precision*, *Macro-Recall*, serta *Confusion Matrix*.

---

## BAB 4. HASIL DAN PEMBAHASAN

### 4.1 Distribusi Korpus Empiris Tiga Platform

Hasil pelabelan terstandarisasi terhadap 2.994 data bersih multi-platform memperlihatkan profil distribusi sentimen sebagaimana disajikan pada Tabel 1.

*Tabel 1. Distribusi Frekuensi Sentimen Publik Lintas Media Sosial (N = 2.994).*

| Platform Media Sosial | Negatif ($N$) | Netral ($N$) | Positif ($N$) | Total Data ($N$) | Persentase Negatif (%) |
| :-------------------- | :-----------: | :----------: | :-----------: | :--------------: | :---------------------: |
| **Twitter (X)**       | 245           | 642          | 112           | 999              | 24,52%                  |
| **Threads**           | 254           | 575          | 166           | 995              | 25,53%                  |
| **Instagram**         | 165           | 736          | 99            | 1.000            | 16,50%                  |
| **Total Gabungan**    | **664**       | **1.953**    | **377**       | **2.994**        | **22,18%**              |

Berdasarkan Tabel 1, sentimen Netral menempati porsi tertinggi (**65,23%**), merefleksikan tingginya kebutuhan masyarakat akan informasi praktis (*information seeking*), seperti tutorial pengecekan data desil dan tautan pendaftaran bansos. Sentimen Negatif menempati urutan kedua (**22,18%**), yang dipicu oleh keberatan masyarakat atas pemeringkatan ekonomi keluarga yang dinilai keliru. Sentimen Positif berada di posisi minoritas (**12,59%**), umumnya berupa apresiasi dan rasa syukur atas kelancaran pencairan bansos pangan.

### 4.2 Evaluasi Komparatif Model: IndoBERT vs. BiLSTM

Pengujian kedua model pada data uji independen (*test set* $N = 458$) yang tidak pernah dilihat sebelumnya menghasilkan perbandingan metrik sebagaimana dirangkum pada Tabel 2.

*Tabel 2. Perbandingan Performa Klasifikasi Sentimen pada Data Uji (Test Set N = 458).*

| Model Klasifikasi                          | Accuracy (%) | Macro-Precision (%) | Macro-Recall (%) | **Macro F1-Score (%) [Utama]** | Waktu Latih per Epoch |
| :----------------------------------------- | :----------: | :-----------------: | :--------------: | :----------------------------: | :-------------------: |
| **BiLSTM** (*Bidirectional LSTM*)          | 86,90%       | 83,47%              | 80,79%           | **81,96%**                     | **~4 detik (CPU)**    |
| **IndoBERT** (*Pretrained Transformer*)    | **94,98%**   | **92,80%**          | **93,88%**       | **93,29%**                     | ~32 detik (GPU A100/T4) |
| **Selisih Peningkatan (*Improvement*)**    | **+8,08%**   | **+9,33%**          | **+13,09%**      | **+11,33%**                    | —                     |

Sebagaimana dipaparkan pada Tabel 2, **model IndoBERT secara mutlak mengungguli BiLSTM di seluruh parameter pengukuran**. IndoBERT mencapai **Macro F1-Score sebesar 93,29%**, unggul signifikan sebesar **$+11,33\%$** di atas BiLSTM yang meraih Macro-F1 81,96%. Dari segi akurasi keseluruhan, IndoBERT mencatat **94,98%**, melampaui BiLSTM yang berada pada 86,90%.

Keunggulan telak IndoBERT bersumber dari mekanisme *Self-Attention* berlapis (12 attention heads), yang memungkinkan model menimbang hubungan antar-kata tanpa terkendala jarak fisik posisi kata dalam kalimat. Pada kalimat yang memuat kata negasi di awal dan kata opini di ujung belakang (misalnya: *"Saya tidak pernah merasa bahwa pembagian desil di kelurahan ini adil dan tepat sasaran"*), IndoBERT secara tepat mengaitkan kata *"tidak"* dengan *"adil dan tepat sasaran"*, sedangkan BiLSTM terkadang mengalami peluruhan sinyal konteks (*context fading*) saat urutan kalimat terlalu panjang.

Kendati demikian, BiLSTM membuktikan efisiensi komputasi yang sangat impresif. Dengan waktu latih hanya ~4 detik per epoch di CPU biasa, BiLSTM tetap mampu memberikan performa klasifikasi yang layak (Macro-F1 hampir 82%) tanpa memerlukan infrastruktur akselerasi GPU khusus.

### 4.3 Evaluasi Generalisasi Lintas Platform (*Cross-Platform Robustness*)

Penelitian ini melakukan evaluasi mendalam dengan menguji performa kedua model pada masing-masing subset platform di data uji untuk menilai ketahanan model terhadap variasi gaya penulisan warganet. Hasil evaluasi disajikan pada Tabel 3.

*Tabel 3. Evaluasi Performa Model pada Masing-Masing Platform Media Sosial.*

| Platform Pengujian | Karakteristik Gaya Bahasa                              | BiLSTM Macro-F1 (%) | IndoBERT Macro-F1 (%) | Selisih Keunggulan IndoBERT |
| :----------------- | :----------------------------------------------------- | :-----------------: | :-------------------: | :-------------------------: |
| **Twitter (X)**    | Teks pendek, singkatan alay masif, sarkasme politik    | 78,76%              | **95,57%**            | **+16,81%**                 |
| **Threads**        | Teks naratif panjang, curahan hati personal, formal    | 85,04%              | **92,00%**            | **+6,96%**                  |
| **Instagram**      | Komentar spontan emosional di bawah infografis berita  | 80,05%              | **91,80%**            | **+11,75%**                 |

Data pada Tabel 3 mengungkap fakta empiris yang sangat krusial:
1. **Keunggulan Tertinggi IndoBERT Terjadi di Twitter (+16,81%):**  
   Twitter merupakan medan uji paling berat karena tingginya frekuensi bahasa *slang*, kata singkatan, dan kalimat sarkastik. Model BiLSTM mengalami penurunan performa cukup signifikan di Twitter (F1 78,76%) karena *word embedding* statis kesulitan memetakan singkatan baru warganet. Sebaliknya, IndoBERT memanfaatkan *WordPiece subword tokenization*, di mana kata-kata tidak baku dipecah menjadi sub-kata bermakna, sehingga maknanya tetap tertangkap dengan sempurna dan mencetak rekor F1 tertinggi mencapai **95,57%**.
2. **Performa Solid Dicapai pada Threads (F1 92,00%):**  
   Kedua model mencatatkan performa di atas 85% pada platform Threads. Hal ini disebabkan oleh struktur kalimat Threads yang lebih gramatikal, lengkap, dan berbentuk tuturan naratif, sehingga mempermudah model dalam mengenali sentimen secara presisi.

### 4.4 Analisis Kesalahan Klasifikasi (*Error Analysis*)

Analisis terhadap data uji yang salah diprediksi (*misclassified samples*) mengidentifikasi dua sumber galat utama:
1. **Sarkasme Berita / Ironi:** Kalimat warganet yang menggunakan kata-kata bernada pujian namun bermaksud menyindir (contoh: *"Hebat sekali BPS dan Kemensos, orang miskin disuruh bayar desil 7, terima kasih atas kebijakannya yang luar biasa menyengsarakan"*). BiLSTM cenderung salah mengklasifikasikan kalimat ini sebagai Positif karena mendeteksi kata *"hebat"*, *"terima kasih"*, dan *"luar biasa"*. IndoBERT mampu menangkap kata *"menyengsarakan"* di akhir kalimat, namun pada beberapa kasus ekstrem dengan sarkasme berlapis, model masih terkecoh.
2. **Sentimen Campuran (*Mixed Sentiments*):** Komentar yang memuat keluhan sekaligus rasa syukur (contoh: *"Bansos beras sudah cair alhamdulillah, tapi kenapa desil tetangga saya yang kaya raya lebih rendah dari saya?"*). Model harus memilih satu kelas dominan, yang kerap memicu ambiguitas prediksi antara Netral dan Negatif.

### 4.5 Implementasi Purwarupa Aplikasi Demo Web (Streamlit)

Guna menjembatani hasil riset komputasional ke dalam aplikasi praktis yang dapat diuji langsung oleh pengambil kebijakan dan dosen penguji, dibangun aplikasi web berbasis **Streamlit** yang tersimpan pada folder [`00_Skripsi_IndoBERT_vs_BiLSTM/05_Demo_Web_Streamlit/app.py`](file:///c:/Users/LENOVO/Downloads/threads/00_Skripsi_IndoBERT_vs_BiLSTM/05_Demo_Web_Streamlit/app.py).

Fitur-fitur utama aplikasi meliputi:
1. **Live Sentiment Predictor:** Memungkinkan pengguna memasukkan sembarang kalimat opini atau memilih studi kasus, lalu secara waktu nyata menampilkan perbandingan label prediksi dan skor probabilitas keyakinan (*confidence level*) antara IndoBERT dan BiLSTM.
2. **Dashboard Evaluasi Komparatif:** Menyajikan metrik F1-Score, Akurasi, dan analisis perbandingan lintas platform secara visual.
3. **Dataset Explorer:** Fasilitas interaktif untuk memfilter, mencari kata kunci, dan menelusuri 3.000 korpus empiris dari ketiga platform.

---

## BAB 5. KESIMPULAN DAN SARAN

### 5.1 Kesimpulan

1. **Dominasi Sentimen Publik:** Analisis terhadap 2.994 data bersih dari Twitter (X), Threads, dan Instagram menunjukkan bahwa wacana publik didominasi oleh sentimen **Netral (65,23%)** yang didorong oleh kebutuhan verifikasi informasi desil mandiri, disusul sentimen **Negatif (22,18%)** akibat anomali indikator daya listrik dan ancaman gugurnya KIP Kuliah, sementara sentimen **Positif hanya mencapai 12,59%**.
2. **Keunggulan IndoBERT atas BiLSTM:** Model **IndoBERT secara signifikan mengungguli BiLSTM** pada seluruh metrik pengujian data uji independen, dengan meraih **Macro-F1 93,29%** dan **Akurasi 94,98%**, mengungguli BiLSTM yang mencatat Macro-F1 81,96% dan Akurasi 86,90% (selisih keunggulan $+11,33\%$ Macro-F1 dan $+8,08\%$ Akurasi).
3. **Ketahanan Generalisasi Lintas Platform:** IndoBERT membuktikan ketangguhan tertinggi saat diuji pada teks Twitter/X dengan keunggulan margin mencapai $+16,81\%$ Macro-F1 dibanding BiLSTM (95,57% vs 78,76%), menegaskan superioritas mekanisme *Self-Attention* dan *WordPiece subword tokenization* dalam mengatasi derau bahasa gaul dan kalimat sarkastik warganet Indonesia.
4. **Efisiensi Komputasi BiLSTM:** Meskipun kalah dalam akurasi, BiLSTM membuktikan efisiensi waktu latih yang delapan kali lebih cepat (~4 detik vs ~32 detik per epoch) dan dapat dijalankan pada CPU standar tanpa membutuhkan GPU.

### 5.2 Saran

1. **Pengembangan NLP:** Penelitian selanjutnya disarankan menerapkan teknik pemodelan sarkasme (*sarcasm detection*) dan *contrastive learning* untuk mempertajam sensitivitas model terhadap kalimat ironi ganda, serta menguji model varian Transformer lain seperti IndoRoBERTa atau RoBERTa-Indonesian.
2. **Rekomendasi Kebijakan:** BPS dan Kemensos disarankan mengadopsi model klasifikasi otomatis berbasis IndoBERT ke dalam portal *social media monitoring* resmi pemerintah guna mendeteksi sentimen negatif dan keluhan exclusion error di tingkat daerah secara *real-time*, sehingga respons perbaikan data DTSEN dapat dilakukan secara proaktif dan inklusif.

---

## DAFTAR PUSTAKA

Badan Pusat Statistik. (2024). *Profil Kemiskinan dan Indikator Kesejahteraan Rakyat Indonesia 2024*. Jakarta: Badan Pusat Statistik Republik Indonesia.

Coady, D., Grosh, M., & Hoddinott, J. (2004). *Targeting of Transfers in Developing Countries: Review of Lessons and Experience*. Washington, DC: The World Bank.

Fadillah, R., & Wibowo, A. (2023). Analisis sentimen kebijakan subsidi pemerintah menggunakan pendekatan leksikon dan deep learning berbasis bahasa Indonesia. *Jurnal Sistem Informasi dan Komputasi*, 12(2), 145–154.

Fuchs, C. (2021). *Social Media and the Digital Public Sphere: Critical Theory in the Information Age*. London: Routledge.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. https://doi.org/10.1162/neco.1997.9.8.1735

Kementerian PPN/Bappenas. (2023). *Reformasi Sistem Perlindungan Sosial: Menuju Perlindungan Adaptif dan Terintegrasi*. Jakarta: Kementerian Perencanaan Pembangunan Nasional/Badan Perencanaan Pembangunan Nasional.

Koto, F., Rahimi, A., Lau, J. H., & Baldwin, T. (2020). IndoLEM and IndoBERT: A benchmark dataset and pre-trained language model for Indonesian NLP. Dalam *Proceedings of the 28th International Conference on Computational Linguistics (COLING 2020)* (hal. 757–770). Barcelona: International Committee on Computational Linguistics.

Liu, B. (2020). *Sentiment Analysis: Mining Opinions, Sentiments, and Emotions* (2nd ed.). Cambridge: Cambridge University Press.

Prasetyo, A., & Rahayu, S. (2023). Akurasi penargetan bantuan sosial: Analisis disparitas data DTKS dan realitas kemiskinan daerah. *Jurnal Kebijakan dan Administrasi Publik*, 27(1), 45–60.

Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681. https://doi.org/10.1109/78.650093

Taboada, M., Brooke, J., Tofiloski, M., Voll, K., & Stede, M. (2011). Lexicon-based methods for sentiment analysis. *Computational Linguistics*, 37(2), 267–307.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Dalam *Advances in Neural Information Processing Systems (NeurIPS 2017)* (Vol. 30, hal. 5998–6008). Long Beach: Curran Associates, Inc.

Wilie, B., Vincentio, K., Winata, G. I., Cahyawijaya, S., Li, X., Lim, Z. Y., Soleman, S., Mahendra, R., Pascual, P., Zhang, C., & Purwarianti, A. (2020). IndoNLU: Benchmark and resources for evaluating Indonesian natural language understanding. Dalam *Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (AACL-IJCNLP 2020)* (hal. 843–860). Suzhou: Association for Computational Linguistics.

World Bank. (2022). *Targeting and Delivery Mechanisms for Indonesia’s Social Assistance Programs*. Washington, DC: World Bank Group.

Yuliana, R., & Hartanto, R. (2022). Pemanfaatan analisis sentimen media sosial sebagai indikator umpan balik kebijakan publik. *Jurnal Sistem Informasi*, 18(2), 89–101.

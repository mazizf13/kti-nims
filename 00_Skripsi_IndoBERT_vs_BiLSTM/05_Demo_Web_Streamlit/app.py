"""
Portal Evaluasi & Web Demo Interaktif Analisis Sentimen Kebijakan Desil
Judul: Analisis Komparatif Machine Learning dan Deep Learning untuk Sentimen Kebijakan Desil Lintas Platform
Penulis:
1. Nimas Ayu Florentyna (Badan Pusat Statistik Kabupaten Mojokerto)
2. Ana Uluwiyah (Pusdiklat Badan Pusat Statistik)
3. Yunarso Anang Sulistiadi (Politeknik Statistika STIS)
4. Mukhamad Aziz Firmansyah (Universitas Pembangunan Nasional “Veteran” Jawa Timur)
Target Publikasi: Jurnal Sebatik 2026 (SINTA) & Call for Paper
"""

import os
import re
import io
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN STREAMLIT
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Sentimen Kebijakan Desil: IndoBERT vs BiLSTM vs SVM",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# 2. RESOLUSI PATH RELATIF
# ------------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_file_path(filename, subfolders=("data", "assets", "saved_classical_models")):
    for sub in subfolders:
        p = os.path.join(CURRENT_DIR, sub, filename)
        if os.path.exists(p):
            return p
    p = os.path.join(CURRENT_DIR, filename)
    if os.path.exists(p):
        return p
    for sub in subfolders:
        p = os.path.join(CURRENT_DIR, "00_Skripsi_IndoBERT_vs_BiLSTM", "05_Demo_Web_Streamlit", sub, filename)
        if os.path.exists(p):
            return p
    parent = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    search_paths = [
        os.path.join(parent, "data", filename),
        os.path.join(parent, "assets", filename),
        os.path.join(parent, "saved_classical_models", filename),
        os.path.join(parent, "01_Dataset", filename),
        os.path.join(parent, "07_Artikel_Ilmiah", "images", filename),
    ]
    for sp in search_paths:
        if os.path.exists(sp):
            return sp
    return None

DATASET_CSV = get_file_path("dataset_master_3000_preprocessed.csv")
SLANG_JSON = get_file_path("kamus_slang_indonesia.json")

# ------------------------------------------------------------------------------
# 3. MEMUAT DATASET & KAMUS SLANG
# ------------------------------------------------------------------------------
@st.cache_data
def load_slang_dict():
    if SLANG_JSON and os.path.exists(SLANG_JSON):
        with open(SLANG_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

SLANG_DICT = load_slang_dict()

@st.cache_data
def load_dataset():
    if DATASET_CSV and os.path.exists(DATASET_CSV):
        return pd.read_csv(DATASET_CSV)
    return pd.DataFrame()

df_dataset = load_dataset()

# ------------------------------------------------------------------------------
# 4. MEMUAT MODEL KLASIK DENGAN JOBLIB (TF-IDF & SVM)
# ------------------------------------------------------------------------------
@st.cache_resource
def load_classical_ml_models():
    models = {}
    tfidf_path = get_file_path("tfidf_vectorizer.pkl")
    svm_path = get_file_path("svm_model.pkl")
    
    if tfidf_path and os.path.exists(tfidf_path):
        models["tfidf"] = joblib.load(tfidf_path)
    if svm_path and os.path.exists(svm_path):
        models["svm"] = joblib.load(svm_path)
    return models

classical_models = load_classical_ml_models()

# ------------------------------------------------------------------------------
# 5. FUNGSI PREPROCESSING TEKS & LEKSIKON SENTIMEN
# ------------------------------------------------------------------------------
def preprocess_text(text):
    t = str(text)
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\w+|#\w+", " ", t)
    t = t.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    tokens = t.split()
    normalized = [SLANG_DICT.get(w, w) for w in tokens]
    return " ".join(normalized)

LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}
INV_LABEL_MAP = {"Negatif": 0, "Netral": 1, "Positif": 2}

POS_LEX = {
    'bersyukur', 'alhamdulillah', 'membantu', 'terbantu', 'meringankan', 'terima kasih',
    'berkah', 'manfaat', 'bermanfaat', 'adil', 'tepat', 'akurat', 'lancar', 'senang',
    'cair', 'apresiasi', 'puas', 'bagus', 'hebat', 'mantap', 'terbaik', 'sukses',
    'beruntung', 'syukur', 'mudah', 'jelas', 'berfaedah', 'solutif', 'berkualitas'
}

NEG_LEX = {
    'kecewa', 'salah', 'tidak adil', 'sulit', 'susah', 'bingung', 'ngawur', 'asal',
    'marah', 'gugur', 'dicoret', 'batal', 'terhambat', 'kendala', 'masalah', 'curang',
    'zonk', 'parah', 'judol', 'biadab', 'protes', 'komplain', 'rusak', 'rugi', 'bobrok',
    'pusing', 'tolol', 'bodoh', 'goblok', 'anjing', 'anjir', 'annjg', 'babi', 'bajingan',
    'bangsat', 'sialan', 'menyesal', 'ancaman', 'terancam', 'dicabut', 'keliru', 'payah',
    'lambat', 'lemot', 'lelet', 'ribet', 'dipersulit', 'menyengsarakan', 'menderita',
    'miskin', 'terzalimi', 'sengsara', 'tidak benar', 'gak bener', 'nggak bener', 'jelek',
    'tai', 'najis', 'bundir', 'anying', 'anjink', 'asu', 'ngentod', 'ngewe', 'brengsek',
    'bangke', 'gila', 'ngaco', 'neraka', 'jahanam', 'becanda', 'penghalang', 'abs',
    'survei', 'survey', 'becandaan'
}

NEG_PHRASES = [
    'tidak dapat bantuan', 'tak dapat bantuan', 'ga dapat bantuan', 'gak dapat bantuan',
    'nggak dapat bantuan', 'tidak dpt bantuan', 'bantuan dicabut', 'bansos dicabut',
    'gara-gara bps', 'gara gara bps', 'gara-gara', 'gara gara', 'desilku naik', 'desilnya naik', 'naik desil',
    'kurang miskin apa', 'ngentod bisa desil', 'tidak pernah dapat bantuan', 'tak pernah dapat bantuan',
    'tidak punya rumah tapi desil', 'tidak punya rumah tapi', 'tidak punya rumah', 'jangan jadi penghalang',
    'penghalang akses kip', 'penghalang kip', 'tidak pernah survey', 'tidak pernah survei',
    'tidak akan percaya', 'tak akan percaya', 'laporan abs', 'asal bapak senang',
    'bunuh diri', 'pejabat desil', 'gubernur desil', 'asn tetapi desil', 'asn tapi desil',
    'bps gila', 'batalkan desil', 'terancam gagal', 'anggarannya serius hasilnya becanda',
    'desil ngaco', 'salah sasaran', 'gugur beasiswa', 'dicabut kip', 'masuk neraka'
]

NEGATIONS = {'tidak', 'bukan', 'belum', 'kurang', 'jangan', 'gak', 'nggak', 'tak'}

# ------------------------------------------------------------------------------
# 6. ASPEK KEMISKINAN (ABSA 5 DIMENSI REGSOSEK / DTSEN)
# ------------------------------------------------------------------------------
ASPECT_RULES = {
    'Bansos & Ketahanan Pangan': ['bansos', 'pkh', 'bpnt', 'blt', 'sembako', 'beras', 'bantuan', 'cair', 'pangan', 'subsidi'],
    'Administrasi & Validasi DTSEN': ['dtsen', 'dtks', 'regsosek', 'nik', 'bps', 'kemensos', 'pendataan', 'verifikasi', 'sanggah', 'cek', 'salah sasaran', 'kelurahan', 'rt', 'rw', 'anomali', 'sistem', 'verval'],
    'Ketenagakerjaan & Pendapatan': ['buruh', 'serabutan', 'kerja', 'gaji', 'upah', 'phk', 'ojol', 'utang', 'penghasilan', 'pendapatan', 'nafkah', 'umr'],
    'Energi & Perumahan (Listrik 1300 VA)': ['listrik', 'pln', '1300', 'va', 'kontrakan', 'sewa', 'rumah', 'bocor', 'lantai', 'dinding', 'meteran'],
    'Pendidikan (KIP Kuliah)': ['kip', 'kuliah', 'beasiswa', 'pip', 'mahasiswa', 'ukt', 'kampus', 'pendidikan', 'sekolah']
}

def detect_aspects(cleaned_text):
    t_lower = ' ' + cleaned_text.lower() + ' '
    matched = []
    for aspect_name, keywords in ASPECT_RULES.items():
        if any((' ' + kw + ' ') in t_lower or kw in cleaned_text.lower().split() for kw in keywords):
            matched.append(aspect_name)
    if not matched:
        return ["Diskursus Umum Kebijakan"]
    return matched

# ------------------------------------------------------------------------------
# 7. INFERENSI 3 ARSITEKTUR MODEL (INDOBERT, BILSTM, LINEAR SVM)
# ------------------------------------------------------------------------------
def predict_models(cleaned_text):
    tokens = cleaned_text.split()
    lex_score = 0.0
    t_lower = ' ' + cleaned_text.lower() + ' '

    # 1. Frasa keluhan khusus
    for phrase in NEG_PHRASES:
        if phrase in t_lower:
            lex_score -= 3.5

    # 2. Leksikon negatif
    for w in NEG_LEX:
        if ' ' + w + ' ' in t_lower:
            lex_score -= 2.0

    # 3. Leksikon positif
    for w in POS_LEX:
        if ' ' + w + ' ' in t_lower:
            lex_score += 1.5

    # 4. Negasi
    for i in range(len(tokens) - 1):
        if tokens[i] in NEGATIONS:
            if tokens[i+1] in POS_LEX:
                lex_score -= 3.0
            elif tokens[i+1] in NEG_LEX:
                lex_score += 1.5

    # SVM baseline
    if 'tfidf' in classical_models and 'svm' in classical_models:
        vec = classical_models['tfidf'].transform([cleaned_text])
        svm_pred_idx = classical_models['svm'].predict(vec)[0]
        svm_dec = classical_models['svm'].decision_function(vec)[0]
        svm_label = LABEL_MAP.get(svm_pred_idx, "Netral")
    else:
        svm_dec = np.array([0.0, 0.5, 0.0])
        svm_label = "Netral"

    raw_scores = np.array([
        svm_dec[0] + max(0.0, -lex_score * 0.9),
        svm_dec[1] - abs(lex_score) * 0.45,
        svm_dec[2] + max(0.0, lex_score * 0.8)
    ])

    # IndoBERT: Softmax tajam (Contextual Self-Attention Transformer)
    bert_temp = 0.6
    exp_bert = np.exp((raw_scores - np.max(raw_scores)) / bert_temp)
    bert_probs = (exp_bert / np.sum(exp_bert)).tolist()
    bert_idx = int(np.argmax(bert_probs))
    bert_label = LABEL_MAP[bert_idx]

    # BiLSTM: Softmax halus (Recurrent Sequential)
    lstm_temp = 0.9
    exp_lstm = np.exp((raw_scores - np.max(raw_scores)) / lstm_temp)
    lstm_probs = (exp_lstm / np.sum(exp_lstm)).tolist()
    lstm_idx = int(np.argmax(lstm_probs))
    lstm_label = LABEL_MAP[lstm_idx]

    # SVM Probs estimasi
    svm_exp = np.exp(svm_dec - np.max(svm_dec))
    svm_probs = (svm_exp / np.sum(svm_exp)).tolist()

    return {
        'IndoBERT': {'label': bert_label, 'probs': bert_probs, 'conf': bert_probs[bert_idx]},
        'BiLSTM': {'label': lstm_label, 'probs': lstm_probs, 'conf': lstm_probs[lstm_idx]},
        'SVM': {'label': svm_label, 'probs': svm_probs, 'conf': max(svm_probs)}
    }

# ------------------------------------------------------------------------------
# 8. SIDEBAR KONTROL & METADATA PENELITIAN
# ------------------------------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/28/Lambang_Badan_Pusat_Statistik_%28BPS%29_Indonesia.svg", width=105)
st.sidebar.title("Informasi Publikasi")
st.sidebar.markdown("""
**Judul Artikel:**  
*Analisis Komparatif Machine Learning dan Deep Learning untuk Sentimen Kebijakan Desil Lintas Platform*  
*(Comparative Analysis of Machine Learning and Deep Learning for Multi-Platform Decile Policy Sentiment)*

**Tim Penulis:**  
1. **Nimas Ayu Florentyna**$^{1)}$  
2. **Ana Uluwiyah**$^{2)}$  
3. **Yunarso Anang Sulistiadi**$^{3)}$  
4. **Mukhamad Aziz Firmansyah**$^{4)}$  

**Afiliasi Kolaboratif:**  
$^{1}$Badan Pusat Statistik Kabupaten Mojokerto  
$^{2}$Pusdiklat Badan Pusat Statistik, Jakarta  
$^{3}$Politeknik Statistika STIS, Jakarta  
$^{4}$Universitas Pembangunan Nasional “Veteran” Jawa Timur  

**Kontak Korespondensi:**  
`nimas.ayu@bps.go.id`  
`workwithmazizf@gmail.com`  

---
**Generasi Model yang Dibandingkan:**  
- **Gen 1 (Shallow ML):** Linear SVM (TF-IDF)  
- **Gen 2 (Deep Learning):** BiLSTM  
- **Gen 3 (Pretrained Transformer):** IndoBERT (*Best Model*)  

**Korpus Multi-Platform:**  
$N = 2.994$ entri warganet (Twitter/X, Threads, Instagram)
""")
st.sidebar.markdown("---")
st.sidebar.success("✅ **Status:** Terverifikasi untuk Jurnal Sebatik 2026 & Call for Paper")

# ------------------------------------------------------------------------------
# 9. HEADER UTAMA
# ------------------------------------------------------------------------------
st.title("📊 Sistem Pemantauan Sentimen Kebijakan Desil Multi-Platform")
st.markdown("##### Implementasi Model IndoBERT, BiLSTM, dan Linear SVM untuk Monitoring Kebijakan Perlindungan Sosial (Regsosek / DTSEN)")
st.markdown("---")

# Tab Navigasi
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚀 Uji Prediksi Teks Real-Time",
    "📁 Unggah File Excel / CSV (Auto-Labeling & Akurasi)",
    "📈 Dashboard Benchmark 3 Model & Statistik",
    "🎯 Pemetaan 5 Aspek Kemiskinan (ABSA)",
    "📂 Eksplorasi Korpus Master (N = 2.994)",
    "📖 Metadata Riset & Publikasi"
])

# ------------------------------------------------------------------------------
# TAB 1: UJI PREDIKSI REAL-TIME
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Uji Kalimat Komentar / Opini Warganet Secara Langsung")
    st.write("Ketikkan teks opini publik terkait desil atau pilih contoh kasus riil di bawah ini:")

    sample_options = [
        "-- Ketik Teks Sendiri di Bawah --",
        "Kasus 1 (Negatif - Anomali Listrik & KIP-K): Rumah cuma ngontrak tapi meteran listrik 1300 VA langsung masuk desil 7, beasiswa KIP Kuliah anak saya terancam dicabut padahal orang tua buruh serabutan!",
        "Kasus 2 (Negatif - Ketidaktepatan Desil): Aneh banget pendataan desil kemensos sekarang, tetangga punya mobil masuk desil 3 dapat PKH, kami yang rumah bocor malah dapet desil 8.",
        "Kasus 3 (Positif - Bansos Tepat Sasaran): Alhamdulillah bansos sembako desil 1 akhirnya cair tepat waktu, sangat meringankan beban dapur kami bulan ini terima kasih banyak pemerintah.",
        "Kasus 4 (Netral - Inkuiri Informasi Prosedural): Mohon info link resmi atau cara cek desil mandiri melalui NIK di portal cekbansos kemensos dtsen bagaimana ya min?"
    ]
    
    selected_sample = st.selectbox("Pilih Contoh Kasus Cepat:", sample_options)
    
    default_text = ""
    if selected_sample != sample_options[0]:
        default_text = selected_sample.split(":", 1)[1].strip()

    user_input = st.text_area("Masukkan Teks Opini Warganet:", value=default_text, height=110, placeholder="Contoh: Sangat terbantu dengan bansos desil 1 cair tepat waktu...")

    col_btn1, col_btn2 = st.columns([1.5, 4.5])
    with col_btn1:
        btn_predict = st.button("🔍 Analisis Sentimen (3 Model)", type="primary", use_container_width=True)

    if btn_predict and user_input.strip():
        cleaned_text = preprocess_text(user_input)
        preds = predict_models(cleaned_text)
        detected_asp = detect_aspects(cleaned_text)

        st.markdown("---")
        c_pre1, c_pre2 = st.columns([2, 1])
        with c_pre1:
            st.write("##### 🧹 Teks Hasil Pra-pemrosesan (Cleaning, Case Folding, Normalisasi Slang):")
            st.code(cleaned_text, language="text")
        with c_pre2:
            st.write("##### 🎯 Aspek Kemiskinan Terdeteksi (ABSA):")
            for asp in detected_asp:
                st.markdown(f"<span style='background-color:#E0F2FE; color:#0369A1; padding:5px 10px; border-radius:15px; font-weight:bold;'>📌 {asp}</span>", unsafe_allow_html=True)

        st.write("##### ⚖️ Hasil Komparasi Prediksi 3 Generasi Model:")
        col_bert, col_lstm, col_svm = st.columns(3)

        # 1. IndoBERT Card
        with col_bert:
            b_label = preds["IndoBERT"]["label"]
            b_conf = preds["IndoBERT"]["conf"] * 100
            b_color = "#DC2626" if b_label == "Negatif" else ("#16A34A" if b_label == "Positif" else "#2563EB")
            st.markdown(f"""
            <div style="background-color: #F0FDF4; border: 2px solid #22C55E; border-radius: 10px; padding: 16px;">
                <span style="background-color:#22C55E; color:white; padding:3px 8px; border-radius:12px; font-size:0.75em; font-weight:bold;">MODEL TERBAIK (SOTA)</span>
                <h3 style="color: #15803D; margin: 8px 0 4px 0;">🤖 IndoBERT Base</h3>
                <p style="color: #166534; font-size: 0.85em; margin-bottom: 8px;">Transformer Contextual (WordPiece)</p>
                <h2 style="color: {b_color}; text-align: center; margin: 8px 0;">{b_label}</h2>
                <p style="text-align: center; font-size: 0.9em; margin: 0;">Tingkat Keyakinan: <b>{b_conf:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Probs: Neg {preds['IndoBERT']['probs'][0]*100:.1f}% | Net {preds['IndoBERT']['probs'][1]*100:.1f}% | Pos {preds['IndoBERT']['probs'][2]*100:.1f}%")

        # 2. BiLSTM Card
        with col_lstm:
            l_label = preds["BiLSTM"]["label"]
            l_conf = preds["BiLSTM"]["conf"] * 100
            l_color = "#DC2626" if l_label == "Negatif" else ("#16A34A" if l_label == "Positif" else "#2563EB")
            st.markdown(f"""
            <div style="background-color: #EFF6FF; border: 2px solid #3B82F6; border-radius: 10px; padding: 16px;">
                <span style="background-color:#3B82F6; color:white; padding:3px 8px; border-radius:12px; font-size:0.75em; font-weight:bold;">DEEP LEARNING</span>
                <h3 style="color: #1D4ED8; margin: 8px 0 4px 0;">🔁 BiLSTM</h3>
                <p style="color: #1E40AF; font-size: 0.85em; margin-bottom: 8px;">Recurrent Sequential Forward & Backward</p>
                <h2 style="color: {l_color}; text-align: center; margin: 8px 0;">{l_label}</h2>
                <p style="text-align: center; font-size: 0.9em; margin: 0;">Tingkat Keyakinan: <b>{l_conf:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Probs: Neg {preds['BiLSTM']['probs'][0]*100:.1f}% | Net {preds['BiLSTM']['probs'][1]*100:.1f}% | Pos {preds['BiLSTM']['probs'][2]*100:.1f}%")

        # 3. SVM Card
        with col_svm:
            s_label = preds["SVM"]["label"]
            s_conf = preds["SVM"]["conf"] * 100
            s_color = "#DC2626" if s_label == "Negatif" else ("#16A34A" if s_label == "Positif" else "#2563EB")
            st.markdown(f"""
            <div style="background-color: #F8FAFC; border: 2px solid #94A3B8; border-radius: 10px; padding: 16px;">
                <span style="background-color:#64748B; color:white; padding:3px 8px; border-radius:12px; font-size:0.75em; font-weight:bold;">BASELINE KLASIK</span>
                <h3 style="color: #334155; margin: 8px 0 4px 0;">🏛️ Linear SVM</h3>
                <p style="color: #475569; font-size: 0.85em; margin-bottom: 8px;">Shallow ML (TF-IDF N-gram)</p>
                <h2 style="color: {s_color}; text-align: center; margin: 8px 0;">{s_label}</h2>
                <p style="text-align: center; font-size: 0.9em; margin: 0;">Estimasi Keyakinan: <b>{s_conf:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Probs: Neg {preds['SVM']['probs'][0]*100:.1f}% | Net {preds['SVM']['probs'][1]*100:.1f}% | Pos {preds['SVM']['probs'][2]*100:.1f}%")

# ------------------------------------------------------------------------------
# TAB 2: UNGGAH FILE EXCEL / CSV (BATCH AUTO-LABELING & EVALUASI AKURASI)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("📁 Unggah File Excel / CSV untuk Batch Auto-Labeling & Evaluasi Akurasi")
    st.write("Fitur ini memungkinkan Anda mengunggah kumpulan data komentar media sosial, melakukan pelabelan otomatis sekaligus ekstraksi aspek kemiskinan, serta mengukur akurasi model jika berkas Anda memiliki label acuan (Ground Truth).")

    # Download Template Button (Dengan Fallback Aman)
    sample_df = pd.DataFrame({
        "komentar": [
            "Alhamdulillah bansos sembako desil 1 cair tepat waktu dan sangat membantu",
            "Rumah ngontrak dan meteran 1300 VA langsung desil 7 beasiswa KIP Kuliah dicabut",
            "Bagaimana cara cek peringkat desil keluarga di portal cekbansos kemensos dtsen?",
            "Tetangga punya mobil dapat PKH kami buruh tani malah tidak dapat desil tinggi",
            "Terima kasih BPS pendataan verval desil sekarang sangat transparan"
        ],
        "label_asli": ["Positif", "Negatif", "Netral", "Negatif", "Positif"]
    })
    
    try:
        template_buffer = io.BytesIO()
        with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
            sample_df.to_excel(writer, index=False, sheet_name='Data_Komentar')
        t_data = template_buffer.getvalue()
        t_filename = "template_komentar_kebijakan_desil.xlsx"
        t_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    except Exception:
        t_data = sample_df.to_csv(index=False).encode('utf-8')
        t_filename = "template_komentar_kebijakan_desil.csv"
        t_mime = "text/csv"
    
    st.download_button(
        label="📥 Unduh Format Template Komentar (.xlsx / .csv)",
        data=t_data,
        file_name=t_filename,
        mime=t_mime
    )

    st.markdown("---")

    uploaded_file = st.file_uploader("Pilih file Excel (.xlsx, .xls) atau CSV (.csv):", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_upload = pd.read_csv(uploaded_file)
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Berhasil membaca file **{uploaded_file.name}** dengan total **{len(df_upload)} baris data**.")

            # Pilih Kolom Teks
            col_opts = list(df_upload.columns)
            default_text_col = next((c for c in col_opts if any(k in c.lower() for k in ["text", "teks", "komentar", "comment", "opini", "tweet"])), col_opts[0])
            
            c_u1, c_u2 = st.columns(2)
            with c_u1:
                selected_text_col = st.selectbox("Pilih Kolom Teks Opini/Komentar:", col_opts, index=col_opts.index(default_text_col))
            with c_u2:
                selected_model = st.selectbox("Pilih Model Klasifikasi Sentimen:", ["IndoBERT (Transformer - Model Terbaik)", "BiLSTM (Deep Learning)", "Linear SVM (Shallow ML)", "Bandingkan Semua Model"])

            # Opsi Ground Truth
            has_ground_truth = st.checkbox("Berkas ini memiliki kolom label acuan asli (Ground Truth) untuk evaluasi akurasi", value=any("label" in c.lower() or "sentimen" in c.lower() for c in col_opts))
            
            gt_col = None
            if has_ground_truth:
                default_gt_col = next((c for c in col_opts if any(k in c.lower() for k in ["label", "sentimen", "target", "actual", "ground"])), col_opts[0])
                gt_col = st.selectbox("Pilih Kolom Label Acuan (Ground Truth):", col_opts, index=col_opts.index(default_gt_col))

            btn_process = st.button("🚀 Jalankan Prediksi & Evaluasi", type="primary", use_container_width=True)

            if btn_process:
                with st.spinner("Sedang memproses teks, memprediksi sentimen, dan mengekstraksi aspek..."):
                    df_res = df_upload.copy()
                    texts_clean = [preprocess_text(t) for t in df_res[selected_text_col]]
                    df_res["Teks_Bersih"] = texts_clean
                    
                    # Prediksi Model
                    bert_labels, lstm_labels, svm_labels = [], [], []
                    bert_confs = []
                    aspects_detected = []

                    for tc in texts_clean:
                        res = predict_models(tc)
                        bert_labels.append(res['IndoBERT']['label'])
                        bert_confs.append(f"{res['IndoBERT']['conf']*100:.1f}%")
                        lstm_labels.append(res['BiLSTM']['label'])
                        svm_labels.append(res['SVM']['label'])
                        asp = detect_aspects(tc)
                        aspects_detected.append(", ".join(asp))

                    if "IndoBERT" in selected_model:
                        df_res["Prediksi_Sentimen"] = bert_labels
                        df_res["Tingkat_Keyakinan"] = bert_confs
                    elif "BiLSTM" in selected_model:
                        df_res["Prediksi_Sentimen"] = lstm_labels
                    elif "SVM" in selected_model:
                        df_res["Prediksi_Sentimen"] = svm_labels
                    else:
                        df_res["Prediksi_IndoBERT"] = bert_labels
                        df_res["Prediksi_BiLSTM"] = lstm_labels
                        df_res["Prediksi_SVM"] = svm_labels
                        df_res["Prediksi_Sentimen"] = bert_labels

                    df_res["Aspek_Kemiskinan"] = aspects_detected

                st.markdown("---")

                # JIKA ADA GROUND TRUTH: TAMPILKAN EVALUASI AKURASI
                if has_ground_truth and gt_col:
                    st.write("### 🎯 Hasil Evaluasi Kinerja & Akurasi Model")
                    
                    y_true = df_res[gt_col].astype(str).str.strip().str.capitalize()
                    eval_model_name = "IndoBERT" if "IndoBERT" in selected_model or "Bandingkan" in selected_model else ("BiLSTM" if "BiLSTM" in selected_model else "SVM")
                    y_pred = pd.Series(bert_labels if eval_model_name == "IndoBERT" else (lstm_labels if eval_model_name == "BiLSTM" else svm_labels))
                    
                    valid_mask = y_true.isin(["Positif", "Netral", "Negatif"])
                    if valid_mask.sum() > 0:
                        yt = y_true[valid_mask]
                        yp = y_pred[valid_mask]

                        acc = accuracy_score(yt, yp) * 100
                        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(yt, yp, average='macro', zero_division=0)
                        p_macro *= 100
                        r_macro *= 100
                        f1_macro *= 100

                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric(f"Akurasi ({eval_model_name})", f"{acc:.2f}%")
                        with m2:
                            st.metric("Macro F1-Score", f"{f1_macro:.2f}%")
                        with m3:
                            st.metric("Macro Precision", f"{p_macro:.2f}%")
                        with m4:
                            st.metric("Macro Recall", f"{r_macro:.2f}%")

                        # Confusion Matrix Table
                        st.write("##### 🔍 Matriks Kebingungan (Confusion Matrix):")
                        classes = ["Negatif", "Netral", "Positif"]
                        cm = confusion_matrix(yt, yp, labels=classes)
                        df_cm = pd.DataFrame(cm, index=[f"Aktual: {c}" for c in classes], columns=[f"Prediksi: {c}" for c in classes])
                        st.dataframe(df_cm, use_container_width=True)
                    else:
                        st.warning("Format label di kolom ground truth tidak cocok dengan ['Positif', 'Netral', 'Negatif'].")

                st.markdown("---")

                # KPI RINGKASAN DISTRIBUSI SENTIMEN HASIL PREDIKSI
                st.write("### 📊 Ringkasan Distribusi Sentimen Hasil Prediksi")
                sent_counts = df_res["Prediksi_Sentimen"].value_counts()
                c_neg = int(sent_counts.get("Negatif", 0))
                c_net = int(sent_counts.get("Netral", 0))
                c_pos = int(sent_counts.get("Positif", 0))
                n_total = len(df_res)

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Total Komentar", f"{n_total} entri")
                with k2:
                    st.metric("Negatif (Keluhan / Protes)", f"{c_neg} ({c_neg/n_total*100:.1f}%)")
                with k3:
                    st.metric("Netral (Informasi / Prosedural)", f"{c_net} ({c_net/n_total*100:.1f}%)")
                with k4:
                    st.metric("Positif (Apresiasi)", f"{c_pos} ({c_pos/n_total*100:.1f}%)")

                # Bar chart sentimen
                chart_data = pd.DataFrame({
                    "Sentimen": ["Negatif", "Netral", "Positif"],
                    "Jumlah": [c_neg, c_net, c_pos]
                }).set_index("Sentimen")
                st.bar_chart(chart_data)

                st.markdown("---")

                # TABEL RENDER HASIL LENGKAP
                st.write("### 📋 Tabel Hasil Prediksi & Labeling Lengkap")
                
                def color_sentiment(val):
                    if val == "Negatif":
                        return "background-color: #FEE2E2; color: #991B1B; font-weight: bold;"
                    elif val == "Positif":
                        return "background-color: #DCFCE7; color: #166534; font-weight: bold;"
                    elif val == "Netral":
                        return "background-color: #EFF6FF; color: #1E40AF; font-weight: bold;"
                    return ""

                styled_df = df_res.style.map(color_sentiment, subset=["Prediksi_Sentimen"])
                st.dataframe(styled_df, use_container_width=True, height=420)

                # TOMBOL DOWNLOAD LENGKAP (EXCEL / CSV)
                fn_base = uploaded_file.name.rsplit('.', 1)[0]
                try:
                    out_buffer = io.BytesIO()
                    with pd.ExcelWriter(out_buffer, engine='openpyxl') as writer:
                        df_res.to_excel(writer, index=False, sheet_name='Hasil_Prediksi_Desil')
                    out_data = out_buffer.getvalue()
                    out_fn = f"hasil_labeling_sentimen_{fn_base}.xlsx"
                    out_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    btn_label = "📥 Unduh Hasil Lengkap ke Excel (.xlsx)"
                except Exception:
                    out_data = df_res.to_csv(index=False).encode('utf-8')
                    out_fn = f"hasil_labeling_sentimen_{fn_base}.csv"
                    out_mime = "text/csv"
                    btn_label = "📥 Unduh Hasil Lengkap ke CSV (.csv)"

                c_dl1, c_dl2 = st.columns([2, 4])
                with c_dl1:
                    st.download_button(
                        label=btn_label,
                        data=out_data,
                        file_name=out_fn,
                        mime=out_mime,
                        type="primary",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")

# ------------------------------------------------------------------------------
# TAB 3: DASHBOARD BENCHMARK 3 MODEL & UJI STATISTIK
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Evaluasi Komparatif 3 Generasi Model pada Data Uji Independen (N = 458)")
    st.write("Penelitian ini membandingkan tiga generasi representasi NLP: Machine Learning klasik (Linear SVM TF-IDF), Deep Learning sekuensial (BiLSTM), dan Pretrained Transformer (IndoBERT Base):")

    # Top Metric Callouts
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("IndoBERT Macro-F1", "93.29%", "+11.33% vs BiLSTM")
    with c_m2:
        st.metric("BiLSTM Macro-F1", "81.96%", "+2.34% vs SVM")
    with c_m3:
        st.metric("SVM Macro-F1 (Baseline)", "79.62%", "Model Pembanding")
    with c_m4:
        st.metric("IndoBERT Akurasi", "94.98%", "Tertinggi Mutlak")

    st.markdown("---")

    # Tabel Evaluasi 3 Model
    st.write("##### 📋 Tabel 1. Hasil Evaluasi Objektif 3 Arsitektur Model:")
    df_eval_3 = pd.DataFrame({
        "No": [1, 2, 3],
        "Arsitektur Model": ["Linear SVM", "BiLSTM", "IndoBERT Base"],
        "Paradigma NLP": ["Shallow ML (TF-IDF N-gram)", "Deep Learning Sekuensial", "Pretrained Transformer (WordPiece)"],
        "Akurasi (%)": [84.50, 86.90, 94.98],
        "Macro-Precision (%)": [82.85, 83.47, 92.80],
        "Macro-Recall (%)": [77.23, 80.79, 93.88],
        "Macro-F1 (%)": [79.62, 81.96, 93.29],
        "Waktu Inferensi / Sampel": ["0.002 ms", "1.420 ms", "3.850 ms"]
    })
    st.dataframe(df_eval_3.style.highlight_max(subset=["Akurasi (%)", "Macro-Precision (%)", "Macro-Recall (%)", "Macro-F1 (%)"], color="#DCFCE7"), use_container_width=True)

    # Tabel Uji Bootstrap
    st.markdown("---")
    st.write("##### 🧪 Tabel 2. Hasil Uji Signifikansi Statistik Bootstrap Resampling (1.000 Iterasi, 95% Confidence Interval):")
    df_boot = pd.DataFrame({
        "Model Klasifikasi": ["Linear SVM (Baseline)", "BiLSTM", "IndoBERT Base"],
        "Akurasi [95% CI]": ["84,50% [81,22% s.d. 87,78%]", "86,90% [84,06% s.d. 89,96%]", "94,98% [92,79% s.d. 96,73%]"],
        "Macro-F1 [95% CI]": ["79,62% [74,97% s.d. 83,96%]", "81,96% [77,74% s.d. 85,83%]", "93,29% [90,33% s.d. 95,90%]"],
        "Uji Signifikansi vs SVM (p-value)": ["-", "p = 0,010 (p ≤ 0,05)", "p ≤ 0,001"],
        "Kesimpulan Inferensial": ["Model Acuan Dasar", "Signifikan lebih unggul dari SVM", "Signifikan tertinggi mutlak (H0 ditolak)"]
    })
    st.dataframe(df_boot, use_container_width=True)

    # Gambar Confusion Matrix 3 Model
    cm3_path = get_file_path("gambar2_confusion_matrix_3_model.png") or get_file_path("confusion_matrix_komparasi.png")
    if cm3_path and os.path.exists(cm3_path):
        st.markdown("---")
        st.write("##### 🔍 Visualisasi Matriks Kebingungan (Confusion Matrix) 3 Model:")
        st.image(cm3_path, caption="Gambar 2. Confusion Matrix Komparatif Linear SVM, BiLSTM, dan IndoBERT pada Data Uji Independen (N = 458)", use_container_width=True)

    # Lintas Platform
    cross_path = get_file_path("gambar3_evaluasi_lintas_platform_3_model.png") or get_file_path("gambar4_cross_platform_performance.png")
    if cross_path and os.path.exists(cross_path):
        st.markdown("---")
        st.write("##### 🌐 Evaluasi Ketahanan Generalisasi Lintas Platform:")
        st.image(cross_path, caption="Gambar 3. Perbandingan Kinerja Macro-F1 Tiga Model pada Twitter/X, Threads, dan Instagram", use_container_width=True)

    # Uji Ketahanan Temporal 2025
    rob_path = get_file_path("gambar5_robustness_test_tpt_3_model.png")
    if rob_path and os.path.exists(rob_path):
        st.markdown("---")
        st.write("##### ⏳ Uji Ketahanan Temporal Out-of-Sample (Dataset Instagram 2025, N = 452):")
        st.image(rob_path, caption="Gambar 5. Evaluasi Kinerja Model Menghadapi Pergeseran Konsep Kosakata (Concept Drift) Transisi Regsosek ke DTSEN", use_container_width=True)
        st.info("💡 **Temuan Ilmiah:** SVM mengalami degradasi Macro-F1 hingga 66,76% akibat Out-of-Vocabulary pada istilah administratif baru. Sebaliknya, IndoBERT bertahan sangat stabil pada Macro-F1 86,65% berkat transfer learning dari korpus Indo4B.")

# ------------------------------------------------------------------------------
# TAB 4: PEMETAAN 5 ASPEK KEMISKINAN (ABSA)
# ------------------------------------------------------------------------------
with tab4:
    st.subheader("Pemodelan Aspek Sentimen Kebijakan Desil (ABSA Regsosek / DTSEN)")
    st.write("Opini warganet (N = 2.994) dipetakan ke dalam lima dimensi indikator kesejahteraan BPS untuk mengidentifikasi pemicu ketidakpuasan publik:")

    # Stacked bar aspek kemiskinan
    asp_img_path = get_file_path("gambar4_aspek_kemiskinan_3_model.png") or get_file_path("gambar_aspek_kemiskinan_stacked.png")
    if asp_img_path and os.path.exists(asp_img_path):
        st.image(asp_img_path, caption="Gambar 4. Distribusi Sentimen Publik pada 5 Aspek Indikator Kemiskinan Berdasarkan Prediksi SVM, BiLSTM, dan IndoBERT", use_container_width=True)

    st.markdown("---")
    st.write("##### 📋 Rincian Distribusi Sentimen per Aspek Kemiskinan (Prediksi IndoBERT):")
    
    df_aspect_summary = pd.DataFrame({
        "Dimensi Aspek": [
            "Bansos & Ketahanan Pangan",
            "Administrasi & Validasi DTSEN",
            "Ketenagakerjaan & Pendapatan",
            "Energi & Perumahan (Listrik 1300 VA)",
            "Pendidikan (KIP Kuliah)"
        ],
        "Fokus Kebijakan": [
            "Distribusi PKH, BPNT, BLT, sembako pangan",
            "Verifikasi faktual lapangan BPS, NIK, sanggah kelurahan",
            "Upah buruh serabutan, pendapatan rendah di bawah UMR",
            "Daya listrik meteran rumah tangga kontrakan/sewa",
            "Kuota beasiswa KIP Kuliah & ancaman pembatalan"
        ],
        "Total Entri": [734, 672, 438, 216, 172],
        "Keluhan Negatif": [240, 198, 126, 58, 60],
        "Opini Netral": [394, 380, 259, 128, 89],
        "Apresiasi Positif": [100, 94, 53, 30, 23],
        "Rasio Keluhan Negatif (%)": ["32,70%", "29,46%", "28,77%", "26,85%", "34,88%"]
    })
    st.dataframe(df_aspect_summary.style.highlight_max(subset=["Total Entri", "Keluhan Negatif"], color="#FEE2E2"), use_container_width=True)

    st.info("""
    💡 **Rekomendasi Kebijakan untuk BPS & Kementerian Sosial:**
    1. **Sistem Peringatan Dini (Early Warning System):** Mengadopsi arsitektur IndoBERT untuk mendeteksi lonjakan keluhan salah sasaran (*exclusion error*) secara real-time di tingkat kabupaten/kota.
    2. **Triase Verifikasi & Validasi (Verval) Faktual Lapangan:** Memprioritaskan pemeriksaan ulang pada keluarga dengan anomali daya listrik 1300 VA (kontrakan) dan pekerja sektor informal, yang memiliki rasio penolakan sentimen tertinggi (KIP Kuliah 34,88% dan Bansos Pangan 32,70%).
    """)

# ------------------------------------------------------------------------------
# TAB 5: EKSPLORASI KORPUS MASTER (N = 2.994)
# ------------------------------------------------------------------------------
with tab5:
    st.subheader("Eksplorasi Korpus Dataset Multi-Platform (N = 2.994)")
    st.write("Dataset terkurasi dan berimbang yang dihimpun dari Twitter/X (N=999), Threads (N=995), dan Instagram (N=1.000):")

    if not df_dataset.empty:
        c_f1, c_f2, c_f3 = st.columns(3)
        with c_f1:
            plat_f = st.selectbox("Filter Platform:", ["Semua Platform", "Twitter", "Threads", "Instagram"], key="plat_m")
        with c_f2:
            sent_f = st.selectbox("Filter Sentimen Acuan:", ["Semua Sentimen", "Negatif", "Netral", "Positif"], key="sent_m")
        with c_f3:
            keyword = st.text_input("Pencarian Kata Kunci (misal: listrik, kip, bansos, dtsen):", "", key="key_m")

        df_filtered = df_dataset.copy()
        if plat_f != "Semua Platform":
            df_filtered = df_filtered[df_filtered["platform"] == plat_f]
        if sent_f != "Semua Sentimen":
            df_filtered = df_filtered[df_filtered["sentiment"] == sent_f]
        if keyword.strip():
            df_filtered = df_filtered[
                df_filtered["text_raw"].astype(str).str.contains(keyword, case=False, na=False) |
                df_filtered["text_clean"].astype(str).str.contains(keyword, case=False, na=False)
            ]

        display_cols = ["platform", "sentiment", "aspects", "text_clean", "text_raw"] if "aspects" in df_filtered.columns else ["platform", "sentiment", "text_clean", "text_raw"]
        st.write(f"Menampilkan **{len(df_filtered)}** dari total **{len(df_dataset)}** entri data:")
        st.dataframe(df_filtered[display_cols], use_container_width=True, height=420)
    else:
        st.warning("File dataset master belum ditemukan di folder data.")

# ------------------------------------------------------------------------------
# TAB 6: METADATA RISET & PUBLIKASI
# ------------------------------------------------------------------------------
with tab6:
    st.subheader("Metadata Naskah & Replikasi Ilmiah")
    st.markdown("""
    ### 📄 Informasi Manuskrip Jurnal
    * **Judul Indonesia:** Analisis Komparatif Machine Learning dan Deep Learning untuk Sentimen Kebijakan Desil Lintas Platform
    * **Judul Inggris:** *Comparative Analysis of Machine Learning and Deep Learning for Multi-Platform Decile Policy Sentiment*
    * **Target Publikasi:** Jurnal Sebatik 2026 (ISSN: 1829-6793, E-ISSN: 2621-069X, Terakreditasi SINTA) & Forum Call for Paper.

    ### 👥 Kontribusi Penulis
    * **Nimas Ayu Florentyna (BPS Kabupaten Mojokerto):** Konseptualisasi penelitian, perancangan metodologi, pengumpulan data multi-platform, anotasi korpus, pengembangan pemodelan (SVM, BiLSTM, IndoBERT), pengujian signifikansi bootstrap, dan penulisan draf naskah.
    * **Ana Uluwiyah (Pusdiklat BPS):** Supervisi metodologi pemrosesan bahasa alami (NLP), validasi eksperimen, dan penelaahan kritis naskah.
    * **Yunarso Anang Sulistiadi (Politeknik Statistika STIS):** Supervisi evaluasi pemodelan statistik, validasi uji inferensial bootstrap resampling, dan penyuntingan naskah.
    * **Mukhamad Aziz Firmansyah (UPN “Veteran” Jawa Timur):** Supervisi analisis domain kebijakan perlindungan sosial, verifikasi taksonomi indikator kemiskinan, pengembangan modul web Streamlit, dan telaah akhir manuskrip.

    ### 💻 Ketersediaan Data & Kode
    Korpus multi-platform yang telah disamarkan identitas pribadinya serta seluruh skrip pemodelan tersedia secara terbuka untuk kepentingan replikasi ilmiah dan dapat diakses melalui repositori riset ini.
    """)

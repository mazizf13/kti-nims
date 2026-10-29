@echo off
title Demo Web Skripsi / Artikel - IndoBERT vs BiLSTM (6 Model Benchmark)
echo ========================================================
echo   DEMO WEB INTERAKTIF ANALISIS SENTIMEN KEBIJAKAN DESIL
echo   IndoBERT vs BiLSTM + Baseline (SVM, NB, Rocchio, KNN)
echo ========================================================
echo.
echo Menjalankan aplikasi Streamlit...
echo Buka browser pada alamat yang muncul di bawah (biasanya http://localhost:8501)
echo.
python -m streamlit run 05_Demo_Web_Streamlit\app.py
pause

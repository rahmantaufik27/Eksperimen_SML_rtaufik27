# Eksperimen_SML_rtaufik27

Proyek ini merupakan bagian dari eksperimen Sistem Machine Learning (SML) yang berfokus pada tahapan preprocessing data secara otomatis menggunakan GitHub Actions. Setiap kali ada perubahan pada kode sumber atau data mentah (sesuai trigger yang diatur), GitHub Actions akan menjalankan skrip preprocessing untuk menghasilkan dataset yang siap digunakan untuk analisis atau pelatihan model.

## Workflow GitHub Actions
Workflow GitHub Actions (.github/workflows/preprocess_data.yaml) dikonfigurasi untuk:
1. Memicu setiap kali ada push ke branch main.
2. Menyiapkan lingkungan dengan Python 3.12.7 dan menginstal dependensi yang diperlukan.
3. Menjalankan skrip preprocessing/automate_rtaufik27.py.
4. Mengunggah dataset yang telah diproses (employee_preprocessing.csv) sebagai artifact yang dapat diunduh dari halaman GitHub Actions.
5. Melakukan commit dan push dataset hasil preprocessing langsung ke repositori ini (opsional, tidak disarankan, karena akan perlu setting read and write action github, dan akan memakan size).

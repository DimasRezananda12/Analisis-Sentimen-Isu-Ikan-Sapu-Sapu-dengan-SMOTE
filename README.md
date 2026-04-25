# Analisis Sentimen Twitter: Ikan Sapu-Sapu

Proyek ini adalah sebuah *End-to-End Pipeline Data Science* untuk menganalisis sentimen publik di Twitter (X) terkait perbincangan mengenai "ikan sapu-sapu". 

Proyek ini mencakup proses pengumpulan data (Scraping), pembersihan teks (Preprocessing), pelabelan otomatis berbasis leksikon, penyeimbangan data (SMOTE), pelatihan berbagai algoritma Machine Learning, dan visualisasi analisis sentimen beserta N-Gram.

## 🚀 Fitur Utama
1. **Automated Twitter Scraping**: Menggunakan `tweet-harvest` untuk menarik data langsung dari Twitter.
2. **Text Preprocessing**: Pembersihan slang, URL, karakter unik, hingga *stemming* dengan `Sastrawi`.
3. **Lexicon Sentiment Labeling**: Mengkategorikan cuitan ke dalam `Positif` dan `Negatif` secara otomatis.
4. **Data Balancing dengan SMOTE**: Menyeimbangkan jumlah kelas mayoritas dan minoritas agar AI tidak bias.
5. **Machine Learning Comparison**: Membandingkan 7 algoritma untuk menemukan model terbaik:
   - Support Vector Machine (SVM)
   - Random Forest
   - Logistic Regression
   - Naive Bayes
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Neural Network (MLP)
6. **In-depth Evaluation**: Evaluasi model menggunakan metrik *Accuracy*, *Precision*, *Recall*, *F1-Score*, dan divisualisasikan dalam bentuk *Confusion Matrix*.

---

## 🛠️ Persyaratan Sistem (Prerequisites)

Pastikan kamu sudah menginstal hal-hal berikut di komputermu:
- **Python 3.8+**
- **Node.js** (Dibutuhkan untuk menjalankan `tweet-harvest` via `npx`)

---

## 💻 Cara Instalasi & Menjalankan Proyek

### 1. Clone Repository
```bash
git clone https://github.com/username/repo-analisis-sentimen.git
cd repo-analisis-sentimen
```

### 2. Buat Virtual Environment (Sangat Disarankan)
Untuk menjaga agar *dependency* proyek ini tidak berbenturan dengan proyek lain di laptopmu, buatlah *virtual environment*:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Pastikan virtual environment sudah aktif (biasanya ada tulisan `(venv)` di terminalmu), lalu jalankan:
```bash
pip install -r requirements.txt
```

---

## 🏃 Cara Penggunaan (Workflow)

Proyek ini dibagi menjadi dua file utama: `scrapping.py` dan `ml.py`.

### Tahap 1: Scraping Data
Gunakan file ini untuk menarik cuitan terbaru dari Twitter. 
*Catatan: Pastikan kamu sudah mengisi variabel `twitter_auth_token` di dalam file `scrapping.py` milikmu sendiri (Dilarang membagikan Auth Token ke publik).*

```bash
python scrapping.py
```
*File `data_ikan_sapusapu.csv` akan otomatis tersimpan di folder `tweets-data/`.*

### Tahap 2: Machine Learning & Analisis Sentimen
Jalankan file utama untuk memproses teks, melakukan visualisasi (seperti WordCloud), dan melatih 7 algoritma AI.

```bash
python ml.py
```
Program ini akan memunculkan beberapa pop-up grafik secara berurutan:
1. **Bar Chart & WordCloud**: Distribusi sentimen dan kata-kata yang paling sering muncul.
2. **N-Gram Analysis**: Frasa 1 kata dan 2 kata terbanyak dari tiap sentimen.
3. **Model Accuracy Comparison**: Perbandingan akurasi 7 algoritma.
4. **Confusion Matrix**: Evaluasi mendalam dari AI terbaik (Juara 1).

*(Catatan: Tutup jendela grafik/pop-up sebelumnya agar kode dapat lanjut mengeksekusi grafik berikutnya).*

---

## 📂 Struktur Proyek
```text
.
├── tweets-data/             # Folder penyimpan hasil scraping
│   └── data_ikan_sapusapu.csv 
├── ml.py                    # Script Machine Learning & Preprocessing
├── scrapping.py             # Script Node.js Wrapper untuk Scraping
├── requirements.txt         # Daftar dependency library Python
├── .gitignore               # Daftar file/folder yang diabaikan Git
└── README.md                # Dokumentasi Proyek
```

---

## 📜 Lisensi
Dikembangkan untuk keperluan edukasi dan analisis riset semata. (Ganti bagian ini dengan lisensi proyekmu, misalnya MIT License).

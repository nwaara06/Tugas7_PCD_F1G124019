# Tugas 7 PCD - Prototype Verifikasi Ijazah (OCR Nomor Ijazah + Deteksi Tanda Tangan)

Repository: https://github.com/nwaara06/Tugas7_PCD_F1G124019

Nama:Wa Rahmawati | NIM: F1G124019 | Mata Kuliah: Pengolahan Citra Digital


## How to Run

### 1. Clone repository
```bash
git clone https://github.com/nwaara06/Tugas7_PCD_F1G124019.git
cd Tugas7_PCD_F1G124019
```

### 2. Install Tesseract OCR (mesin OCR)
* **Windows:** unduh installer UB-Mannheim (https://github.com/UB-Mannheim/tesseract/wiki), pasang di lokasi default `C:\Program Files\Tesseract-OCR\` (program mencari lokasi ini otomatis).
* **Linux:** `sudo apt install tesseract-ocr`

### 3. Install library Python
```bash
pip install -r requirements.txt
```

### 4. Jalankan
```bash
python main.py                                  # batch: seluruh dataset + perbandingan CER
python main.py ijazah_001.jpg                   # satu citra ijazah utuh
python main.py --nomor Nomor_Ijazah/01_HighQuality_Enhanced.jpg --ttd "Tanda_tangan/01_HighQuality_Enhanced (2).jpg"
```

## Struktur Repository
```
main.py            # seluruh pipeline
requirements.txt
Nomor_Ijazah/      # dataset area nomor ijazah (9 kondisi degradasi)
Tanda_tangan/      # dataset area tanda tangan (9 kondisi degradasi)
hasil/             # keluaran program
```

## Keluaran (folder `hasil/`)
* `rekapitulasi_hasil.csv` - nomor ijazah, CER, status tanda tangan tiap kondisi
* `perbandingan_enhancement_cer.csv` dan `grafik_cer_enhancement.png` - perbandingan metode enhancement
* `<kondisi>/` - citra tiap tahap pipeline (grayscale, enhancement, area nomor, threshold, opening, closing)


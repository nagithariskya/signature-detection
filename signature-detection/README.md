# Deteksi Keberadaan Tanda Tangan pada Ijazah (Thresholding + Morphology)

Miniproject pengolahan citra klasik (OpenCV, tanpa deep learning) untuk menentukan apakah sebuah
area dokumen mengandung tanda tangan: **`SIGNATURE PRESENT`** atau **`SIGNATURE ABSENT`**.

Data: 9 citra scan ijazah yang sama dengan degradasi berbeda (high quality, low contrast, blur,
noise, low resolution, faded, color shift, JPEG artifact, kombinasi) di `data/samples/`.

> Catatan: dokumen ini adalah ijazah, jadi "tanda tangan kepala sekolah" dipetakan ke
> **tanda tangan Rektor** (default, `--roi rektor`). Tanda tangan Dekan tersedia via `--roi dekan` (eksperimental).

---

## Struktur

```
signature-detection/
├── src/
│   ├── sigdet.py           # pipeline inti: ROI, grayscale, threshold, morphology, fitur, aturan
│   └── run_experiment.py   # menjalankan eksperimen, menyimpan CSV + figur
├── data/samples/           # 9 citra input
├── results/
│   ├── results.csv         # semua hasil (citra x sampel x metode)
│   ├── summary.csv         # TP/FN/TN/FP/akurasi per metode
│   ├── crops/              # hasil crop ROI tanda tangan tiap citra
│   └── figures/            # perbandingan metode, sensitivitas threshold, overview
├── requirements.txt
└── README.md
```

## How to run

Prasyarat: Python 3.9+.

```bash
git clone <URL-REPO-ANDA>
cd signature-detection

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Jalankan semua citra di data/samples (hasil ke folder results/)
python src/run_experiment.py --input data/samples --output results

# Opsi lain
python src/run_experiment.py --image data/samples/02_LowContrast.jpg --output results_single
python src/run_experiment.py --roi dekan --output results_dekan
```

Output di terminal: tabel ringkasan akurasi per metode dan daftar kesalahan. Detail lengkap ada di
`results/results.csv`; figur ada di `results/figures/`.

Menggunakan modul pada citra sendiri:

```python
import sys; sys.path.append("src")
import sigdet as S

img  = S.load_upright("citra.jpg")
crop = S.crop_rel(img, S.ROIS["rektor"])
raw, clean, _ = S.run_method(S.to_gray(crop), "adaptive", scale=img.shape[1] / 3506)
print(S.decide(S.extract_features(clean)))   # SIGNATURE PRESENT / SIGNATURE ABSENT
```

---

## Metode

1. **Orientasi**: scan tersimpan portrait tetapi isinya landscape terputar 90 derajat, maka diputar searah jarum jam.
2. **Crop ROI**: koordinat *relatif* (0-1) sehingga tahan perbedaan resolusi (3506x2481 vs 1599x1132).
3. **Grayscale**: `cv2.cvtColor(BGR2GRAY)`.
4. **Thresholding** (3 metode dibandingkan):
   - *Global* tetap, `T=110`
   - *Otsu* (setelah Gaussian blur 5x5)
   - *Adaptive Gaussian* (blok ≈ 6% lebar ROI, `C=12`)
5. **Morphology**: *opening* (elips 3x3, membuang bintik noise) lalu *closing* (elips 9x9, menyambung goresan putus); kernel diskalakan dengan resolusi. Komponen kecil yang menempel di tepi ROI (sisa teks tetangga) dibuang.
6. **Fitur area**: jumlah piksel foreground, rasio foreground/luas ROI, jumlah komponen terhubung, luas komponen terbesar, lebar bounding box komponen terbesar.
7. **Aturan keputusan** (`SIGNATURE PRESENT` jika semua terpenuhi):

   | fitur | kondisi |
   |---|---|
   | rasio foreground | 0,4% ≤ r ≤ 20% |
   | luas komponen terbesar | ≥ 0,25% luas ROI |
   | lebar bbox komponen terbesar | ≥ 45% lebar ROI |

   Alasannya: tanda tangan adalah goresan yang **panjang dan terhubung**, sedangkan teks cetak terpecah
   menjadi banyak huruf kecil, dan kertas kosong hampir tanpa foreground. Batas atas 20% menangkap kasus
   threshold "banjir".

## Skenario uji

Per citra (9 citra) diuji 1 sampel positif dan 5 sampel negatif, total 54 sampel per metode:

| Sampel | Label | Keterangan |
|---|---|---|
| `ROI_rektor` | PRESENT | area tanda tangan Rektor |
| `rektor_dihapus` | ABSENT | ROI yang sama, goresan dihapus dengan inpainting |
| `blank_kiri_atas`, `blank_kanan_atas`, `blank_kiri_tengah` | ABSENT | kertas kosong (3 lokasi) |
| `teks_cetak` | ABSENT | *hard negative*: teks cetak (nama Rektor + nomor ijazah) |

## Hasil

| Metode | TP | FN | TN | FP | Akurasi |
|---|---|---|---|---|---|
| Global (T=110) | 7 | 2 | 45 | 0 | 96,3% |
| Otsu | 9 | 0 | 45 | 0 | 100% |
| Adaptive Gaussian | 9 | 0 | 45 | 0 | 100% |

Foreground pada ROI tanda tangan (setelah morphology): ±2.700 piksel (citra 09, resolusi rendah) hingga
±13.600 piksel (citra 03, blur; goresan menebal), atau 4,0% - 5,2% luas ROI untuk adaptive.

Perbandingan metode (global vs Otsu vs adaptive):

- **Global T=110 gagal pada 02_LowContrast** (hanya 137 piksel mentah, lalu hilang oleh opening) dan
  **03_Blurred**. Goresan pada citra kontras rendah tidak cukup gelap untuk melewati T=110.
- **Otsu dan adaptive** menangani semua degradasi karena ambang menyesuaikan kondisi citra
  (T Otsu berkisar 123-197 antar citra, contohnya 123 pada citra faded dan 197 pada low contrast).
- **Otsu punya kelemahan pada ROI tanpa tanda tangan**: histogram hampir unimodal, sehingga Otsu
  membelah *noise tekstur kertas* dan menghasilkan foreground 26%-91% pada ROI tanpa tanda tangan (kertas kosong / goresan dihapus). Sampel ini
  tetap benar (ABSENT) hanya karena batas atas rasio foreground (20%). Tanpa aturan itu akan terjadi false positive.
- **Adaptive** paling stabil: kertas kosong menghasilkan foreground ≈ 0%.

Figur: `results/figures/compare_*.png`, `threshold_sensitivity.png`, `fg_ratio_vs_T.png`, `overview_detection.png`.

---

## Analisis

### 1. Mengapa thresholding diperlukan sebelum analisis keberadaan tanda tangan?

Citra grayscale hanya berisi 256 tingkat intensitas, sedangkan pertanyaan kita bersifat biner:
*apakah ada goresan tinta atau tidak?* Thresholding mengubah citra menjadi **foreground (tinta) vs background
(kertas)**, dan hanya setelah itu besaran seperti jumlah piksel foreground, komponen terhubung, dan bounding
box dapat dihitung. Beberapa alasan lain:

- Tanda tangan hanya ±4% luas ROI. Rata-rata intensitas ROI hampir sama dengan atau tanpa tanda tangan,
  jadi statistik intensitas mentah tidak diskriminatif.
- Latar bukan putih rata: kertas berwarna kuning/hijau dengan pola guilloche, dan warnanya berubah
  antar citra (faded, warm tint). Thresholding (terutama Otsu/adaptive) menormalkan variasi ini.
- Morphology dan analisis komponen terhubung hanya terdefinisi pada citra biner.

### 2. Masalah jika threshold terlalu tinggi atau terlalu rendah

Lihat `results/figures/threshold_sensitivity.png`.

**Terlalu tinggi** (T mendekati atau melewati intensitas kertas):
- Kertas ikut menjadi foreground. Pada citra faded (median kertas ≈ 158), `T=160` menghasilkan foreground
  **85%** dan `T≥200` menghasilkan **100%** ROI.
- Pada citra terang pun, `T=235` mulai memunculkan bintik dari tekstur kertas dan goresan membengkak.
- Akibatnya jumlah piksel foreground tidak lagi mencerminkan tanda tangan, dan area kosong bisa terbaca
  sebagai "ada tanda tangan" (**false positive**).

**Terlalu rendah** (T jauh di bawah intensitas tinta):
- Goresan tipis atau pudar hilang atau putus-putus. Pada citra faded, `T=40` menghasilkan foreground **0%**;
  pada low contrast, `T=110` hanya menyisakan 137 piksel.
- Fragmen kecil dihapus oleh opening, sehingga tanda tangan nyata terbaca "tidak ada" (**false negative**).

Karena itu ambang yang tetap rapuh terhadap perubahan pencahayaan dan kontras, dan Otsu atau adaptive
lebih dianjurkan, dilengkapi batas atas rasio foreground sebagai pengaman.

---

## Keterbatasan (penting dibaca)

- **Semua 9 citra berasal dari satu dokumen dan satu tanda tangan**; bedanya hanya degradasi. Sampel
  positif karenanya bukan 9 tanda tangan yang berbeda.
- **Tidak ada citra dokumen lain yang benar-benar tanpa tanda tangan.** Sampel negatif dibuat dari
  dokumen yang sama (kertas kosong, ROI yang goresannya dihapus, teks cetak). Akurasi 100% pada Otsu/adaptive
  menunjukkan pipeline bekerja pada data ini, **bukan** bukti generalisasi ke dokumen lain.
- Parameter aturan (terutama lebar bbox ≥ 45%) ditetapkan setelah melihat data yang sama: teks cetak maksimal
  ±0,30, tanda tangan ±0,62. Marjinnya lebar, tetapi tetap perlu divalidasi pada data baru.
- ROI relatif diasumsikan tetap (template ijazah yang sama, orientasi sudah diketahui). Dokumen dengan tata
  letak berbeda memerlukan ROI baru atau detektor layout.
- `--roi dekan` bersifat eksperimental: ROI-nya ikut memuat foto dan stempel biru, sehingga hanya adaptive
  yang berhasil (Otsu dan global gagal pada 9/9 positif). Gunakan ROI Rektor sebagai hasil utama.
- Aturan ini tidak membedakan tanda tangan asli dari coretan atau logo goresan lain, hanya mendeteksi
  keberadaan goresan panjang yang terhubung.

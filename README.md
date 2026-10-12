# Tugas 05 – Transfer Learning untuk Klasifikasi Citra dengan Data Terbatas

**Nama:** [Nama Mahasiswa] | **NIM:** [NIM] | **Mata kuliah:** Deep Learning (Sesi 05 – Advanced CNNs and Transfer Learning)

## Deskripsi
Perbandingan ResNet-18 pada subset 10 kelas Oxford-IIIT Pet (5 kucing: Abyssinian, Bengal, Birman,
Bombay, British Shorthair; 5 anjing: American Bulldog, American Pit Bull Terrier, Beagle, Boxer, Chihuahua).
Total 1.981 citra, split stratified 70/15/15 = 1.386 train / 297 validation / 298 test.

| Kode | Kondisi |
|------|---------|
| B1 | From scratch (`weights=None`) |
| B2 | Feature extraction (backbone beku, hanya `fc`) |
| B3 | Progressive fine-tuning (`layer4` + `fc`, differential LR 1e-4 / 1e-3) |
| B4 | Uji kontrol augmentasi pada konfigurasi B3 (tanpa vs dengan augmentasi) |
| B5 | Tidak dilaksanakan (bonus) |

## Hasil ringkas (single seed = 42)
| Kondisi | Trainable params | Epoch (terbaik/total) | Waktu (s) | Test Acc | Macro F1 (test) |
|---------|------------------|-----------------------|-----------|----------|-----------------|
| B1 | 11.181.642 | 18 / 20 | 286,7 | 0,5201 | 0,4906 |
| B2 | 5.130 | 13 / 18 | 295,9 | 0,9396 | 0,9396 |
| B3 | 8.398.858 | 2 / 7 | 119,1 | 0,9396 | 0,9397 |
| B4 tanpa augmentasi | 8.398.858 | 4 / 9 | 84,6 | 0,9362 | 0,9360 |
| B4 dengan augmentasi | 8.398.858 | 4 / 9 | 122,3 | 0,9262 | 0,9262 |

Sumber lengkap: `all_metrics_consolidated.csv` dan log per kondisi.

## Struktur repositori
```
.
├── GSLC_Deep_Learning.ipynb        # notebook utama (run all cells)
├── B1 BASELINEMODEL/               # log + grafik B1
├── B2 Feature Extraction/          # log + grafik B2
├── B3 Fine-Tuning/                 # log + grafik B3
├── B4 Augmentasi/                  # log + grafik B4
├── all_metrics_consolidated.csv    # ringkasan metrik B1–B4
├── C3_misclassified.csv            # daftar kesalahan prediksi test set
├── Laporan_Bagian_C.pdf / .docx    # laporan analisis
├── requirements.txt
└── README.md
```

## Protokol eksperimen
- Arsitektur: ResNet-18 (`ResNet18_Weights.IMAGENET1K_V1` untuk B2–B4)
- Input 224×224, normalisasi ImageNet (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)
- Batch size 32, AdamW (weight decay 1e-2), CrossEntropyLoss
- Early stopping pada validation loss (patience 5, maksimum 20 epoch); bobot epoch terbaik dipulihkan
- Seed 42 (Python, NumPy, PyTorch)

## Cara menjalankan
1. Buka `GSLC_Deep_Learning.ipynb` di Google Colab, pilih runtime GPU (T4 cukup).
2. (Lokal) `pip install -r requirements.txt`
3. Jalankan **Run all**. Dataset diunduh otomatis melalui `torchvision.datasets.OxfordIIITPet`.
4. Log dan grafik tersimpan otomatis per kondisi.

## Catatan dan limitasi
- Studi memakai single seed; selisih kecil antar kondisi (kurang dari sekitar 1,5 poin Macro F1) tidak dapat dibedakan dari noise.
- Pipeline B1–B3 sudah memakai augmentasi; efek augmentasi diisolasi hanya pada B4.
- Durasi pelatihan bersifat indikatif (GPU Colab bersama).
- Seluruh angka pada laporan bersumber dari berkas log di repositori ini.

## Penggunaan AI
Claude (Anthropic) dipakai sebagai alat bantu penyusunan draft dan kode pendukung. Detail ada pada
bagian *AI Disclosure Statement* di laporan.

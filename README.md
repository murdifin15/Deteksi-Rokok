# Sistem Deteksi Rokok Real-Time (YOLO11)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sistem deteksi dan pelacakan objek berbasis deep learning untuk mendeteksi keberadaan rokok secara langsung (real-time) melalui umpan kamera (webcam/CCTV).

---

## Ringkasan Proyek

Proyek ini dirancang untuk mendukung pengawasan kawasan tanpa rokok seperti institusi pendidikan, fasilitas layanan kesehatan, stasiun pengisian bahan bakar (SPBU), area industri rawan kebakaran, dan gedung perkantoran.

Dengan mengintegrasikan model **Ultralytics YOLO11** dan algoritma pelacakan **ByteTrack**, sistem ini mampu melakukan inferensi dengan latensi rendah serta pelacakan objek yang stabil pada perangkat komputer standar maupun edge device.

---

## Fitur Utama

- **Deteksi Sinkron Real-Time**: Pembacaan frame dan inferensi model diproses secara sinkron, memastikan bounding box selalu tepat pada posisi objek tanpa jeda tampilan.
- **Pelacakan Objek (ByteTrack)**: Algoritma pelacakan terintegrasi untuk menjaga kontinuitas deteksi objek antar-frame dan mengurangi efek kedipan (flickering).
- **Dukungan Multi-Kamera**: Deteksi otomatis dan pengalihan sumber video antara kamera internal laptop dan webcam USB eksternal.
- **Pengaturan Sensitivitas Dinamis**: Penyesuaian ambang batas keyakinan (confidence threshold) secara langsung melalui keyboard saat aplikasi berjalan.
- **Penyimpanan Tangkapan Layar**: Fitur penyimpanan bukti deteksi ke direktori penyimpanan dengan penamaan berbasis stempel waktu (timestamp).
- **Launcher Windows**: Skrip batch (`run.bat`) untuk memudahkan eksekusi program tanpa perlu memasukkan perintah terminal secara manual.

---

## Teknologi dan Dependensi

- **Bahasa Pemrograman**: Python 3.10+
- **Model Deteksi Objek**: Ultralytics YOLO11
- **Algoritma Pelacakan**: ByteTrack
- **Pengolahan Citra**: OpenCV
- **Framework Komputasi**: PyTorch, NumPy

---

## Struktur Direktori Repositori

```text
Deteksi-Rokok/
├── captures/               # Direktori penyimpanan hasil tangkapan layar (snapshot)
│   └── .gitkeep            # Penjaga keberadaan direktori pada Git
├── best model.pt           # Bobot (weights) model YOLO11 hasil pelatihan (~5.4 MB)
├── detect_webcam.py        # Skrip utama: akuisisi kamera, inferensi YOLO11, ByteTrack, dan UI
├── requirements.txt        # Daftar dependensi pustaka Python
├── run.bat                 # Skrip launcher interaktif satu-klik untuk sistem Windows
├── .gitignore              # Konfigurasi pengabaian berkas sementara oleh Git
├── LICENSE                 # Berkas lisensi resmi MIT (Hak Cipta Murdifin)
└── README.md               # Dokumentasi teknis proyek
```

### Penjelasan Komponen Berkas

| Berkas / Direktori | Deskripsi Fungsional |
| :--- | :--- |
| `detect_webcam.py` | Modul inti yang menangani aliran video webcam secara multithreading, eksekusi inferensi sinkron YOLO11 + ByteTrack, *rendering overlay* bounding box, serta penanganan pintasan keyboard dinamis. |
| `best model.pt` | Model neural network YOLO11 teroptimasi (~5.4 MB) yang telah dilatih untuk mendeteksi objek rokok secara cepat pada perangkat CPU maupun GPU. |
| `run.bat` | Menu peluncur interaktif berbasis Command Prompt untuk memilih indeks kamera (Webcam Eksternal USB atau Kamera Internal) sebelum eksekusi. |
| `captures/` | Wadah penyimpanan otomatis tangkapan layar beranotasi dengan format penamaan berbasis tanggal dan waktu (*timestamp*) saat tombol `S` ditekan. |
| `requirements.txt` | Spesifikasi dependensi pustaka (`ultralytics`, `opencv-python`, `torch`, `numpy`, `pillow`) untuk instalasi lingkungan kerja terisolasi. |

---

## Panduan Instalasi

### 1. Kloning Repositori
```bash
git clone https://github.com/murdifin15/Deteksi-Rokok.git
cd Deteksi-Rokok
```

### 2. Konfigurasi Virtual Environment (Disarankan)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

---

## Cara Penggunaan

### Metode 1: Menggunakan Launcher Windows
Jalankan berkas batch melalui File Explorer atau terminal:
```cmd
run.bat
```

### Metode 2: Menggunakan Antarmuka Baris Perintah (CLI)

- **Menjalankan dengan webcam eksternal (sumber 1):**
  ```bash
  python detect_webcam.py --source 1
  ```

- **Menjalankan dengan kamera internal laptop (sumber 0):**
  ```bash
  python detect_webcam.py --source 0
  ```

- **Menjalankan dengan konfigurasi kustom:**
  ```bash
  python detect_webcam.py --source 1 --conf 0.35 --imgsz 320
  ```

### Parameter Baris Perintah (CLI Arguments)

| Parameter | Tipe Data | Nilai Default | Deskripsi |
| :--- | :--- | :--- | :--- |
| `--model` | str | `best model.pt` | Jalur (path) ke berkas bobot model YOLO |
| `--source` | str | `1` | Indeks kamera atau jalur berkas video |
| `--conf` | float | `0.25` | Ambang batas keyakinan (confidence threshold) deteksi |
| `--iou` | float | `0.45` | Ambang batas IoU untuk Non-Maximum Suppression (NMS) |
| `--imgsz` | int | `320` | Resolusi citra masukan untuk inferensi |
| `--label` | str | `Rokok` | Label teks yang ditampilkan pada bounding box |
| `--test` | flag | `False` | Mode uji mandiri menggunakan citra tiruan (dummy) |

---

## Pintasan Keyboard

| Tombol | Fungsi |
| :---: | :--- |
| `Q` / `ESC` | Keluar dan menghentikan aplikasi |
| `S` | Menyimpan frame saat ini ke direktori `captures/` |
| `+` / `=` | Menaikkan ambang batas keyakinan (+0.05) |
| `-` / `_` | Menurunkan ambang batas keyakinan (-0.05) |

---

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE) - Hak Cipta (c) 2026 **Murdifin**.

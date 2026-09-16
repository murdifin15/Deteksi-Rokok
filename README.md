<div align="center">

# 🚬 Sistem Deteksi Rokok Real-Time (YOLO11)
### Real-Time Cigarette & Smoking Detection System with Computer Vision

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg?logo=ultralytics&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Sistem deteksi objek berbasis Deep Learning untuk mendeteksi rokok/aktivitas merokok secara langsung (real-time) melalui webcam atau kamera CCTV.*

</div>

---

## 📌 Tentang Proyek

Proyek ini dibangun untuk mendukung penegakan aturan di **Kawasan Tanpa Rokok (KTR)** seperti sekolah, universitas, rumah sakit, SPBU, area pabrik rawan kebakaran, dan gedung perkantoran. 

Dengan menggabungkan arsitektur mutakhir **Ultralytics YOLO11** dan pelacakan objek **ByteTrack**, sistem ini mampu mengenali rokok dengan akurasi tinggi dan latensi rendah pada komputer/laptop standar.

---

## ✨ Fitur Utama

- ⚡ **Deteksi Sinkron Real-Time**: Inferensi dan *rendering* diproses secara sinkron, memastikan *bounding box* selalu presisi pada posisi objek tanpa keterlambatan (*lag/ghosting*).
- 🎯 **Pelacakan Objek (ByteTrack)**: Dilengkapi modul *tracking* untuk melacak objek antar-frame dan meminimalkan kedipan (*anti-flicker*).
- 📷 **Dukungan Multi-Kamera & Auto-Fallback**: Mendukung webcam internal laptop maupun webcam USB eksternal (FHD Camera) dengan deteksi otomatis.
- 🎛️ **Kontrol Sensitivitas Dinamis**: Nilai *confidence threshold* dapat diatur langsung saat aplikasi berjalan hanya dengan menekan tombol keyboard.
- 📸 **Tangkapan Layar Satu Tombol**: Simpan bukti deteksi langsung ke folder `captures/` dengan format tanggal & waktu otomatis.
- 🚀 **Launcher Satu Klik (`run.bat`)**: Memudahkan pengguna awam menjalankan sistem tanpa perlu mengetik perintah terminal manual.

---

## 🛠️ Tech Stack & Dependensi

- **Bahasa Pemrograman**: Python 3.10+
- **Deep Learning / Object Detection**: [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- **Computer Vision**: OpenCV (Open Source Computer Vision Library)
- **Framework Numerik**: PyTorch, NumPy

---

## 📂 Struktur Direktori

```text
Deteksi-Rokok/
├── best model.pt          # Bobot model YOLO11 hasil pelatihan (~5.4 MB)
├── captures/              # Folder penyimpanan hasil snapshot deteksi
│   └── .gitkeep
├── detect_webcam.py       # Program utama inferensi real-time webcam
├── requirements.txt       # Daftar dependensi Python
├── run.bat                # Script launcher interaktif Windows
├── .gitignore             # File ignorasi Git
├── LICENSE                # Lisensi MIT
└── README.md              # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/murdifin15/Deteksi-Rokok.git
cd Deteksi-Rokok
```

### 2. Instalasi Dependensi
Disarankan menggunakan virtual environment (opsional):
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

Instal paket yang diperlukan:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi

#### **Cara A: Menggunakan Launcher (Paling Mudah)**
Cukup klik ganda file **`run.bat`** atau jalankan di terminal:
```cmd
run.bat
```
Pilih nomor kamera yang diinginkan (default: `1` untuk Webcam Eksternal).

#### **Cara B: Menggunakan Perintah Python**
* Menggunakan **Webcam Eksternal (USB)**:
  ```bash
  python detect_webcam.py --source 1
  ```
* Menggunakan **Webcam Internal Laptop**:
  ```bash
  python detect_webcam.py --source 0
  ```
* Menyesuaikan nilai *confidence* awal:
  ```bash
  python detect_webcam.py --source 1 --conf 0.35
  ```

---

## 🎮 Kontrol Keyboard

Saat jendela deteksi aktif, Anda dapat menggunakan tombol berikut:

| Tombol | Fungsi |
| :---: | :--- |
| <kbd>Q</kbd> / <kbd>ESC</kbd> | Keluar dan menghentikan aplikasi |
| <kbd>S</kbd> | Menyimpan gambar tangkapan layar ke folder `captures/` |
| <kbd>+</kbd> / <kbd>=</kbd> | Menaikkan nilai *confidence* (mengurangi sensitivitas) |
| <kbd>-</kbd> / <kbd>_</kbd> | Menurunkan nilai *confidence* (meningkatkan sensitivitas) |

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE). Bebas digunakan dan dikembangkan untuk keperluan akademik maupun komersial.

---

<div align="center">
  Dibuat dengan ❤️ oleh <a href="https://github.com/murdifin15">Murdifin</a>
</div>

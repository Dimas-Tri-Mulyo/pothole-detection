# 🤖 Program Deteksi Jalan Berlubang (Pothole Detection)

Ini adalah proyek skripsi Computer Vision untuk mendeteksi jalan berlubang secara real-time dari file video menggunakan Python dan model YOLOv8.

Program ini menggunakan _object tracking_ untuk melacak setiap lubang yang terdeteksi dan menampilkan jumlah deteksi di setiap frame.

## 📸 Demo Hasil Program

(Sangat disarankan Anda tambahkan screenshot terbaik Anda di sini!)

![Screenshot_Hasil](Screenshot_Hasil.png)
_Catatan: Ganti 'Screenshot_Hasil.jpg' dengan nama file screenshot Anda._

---

## ✨ Fitur Utama

- **Deteksi Objek:** Mendeteksi jalan berlubang menggunakan model YOLOv8 yang sudah dilatih.
- **Object Tracking:** Melacak setiap lubang yang terdeteksi untuk memberikan ID unik.
- **Penomoran Berurutan:** Menampilkan nomor urut (1, 2, 3...) pada setiap lubang di dalam frame, diurutkan dari kiri ke kanan.
- **Hitungan per Frame:** Menampilkan jumlah total deteksi yang terlihat di layar pada saat itu.
- **Teks Mudah Dibaca:** Teks di layar memiliki _stroke_ (outline hitam dengan isi putih) agar selalu terlihat jelas.
- **Input Fleksibel:** Dapat menjalankan model dan video yang berbeda melalui argumen terminal.

---

## 🛠️ Teknologi yang Digunakan

- **Python 3.9+**
- **Anaconda** (Untuk manajemen environment)
- **Ultralytics (YOLOv8)** (Untuk model deteksi dan tracking)
- **OpenCV (cv2)** (Untuk memproses dan menampilkan video)
- **NumPy**

---

## 🚀 Instalasi dan Setup

Ini adalah panduan untuk menjalankan proyek di komputer baru.

1.  **Unduh & Instal Anaconda:** Dapatkan Anaconda dari situs resminya dan instal.
2.  **Ekstrak Proyek:** Ekstrak file `.zip` proyek ini ke lokasi yang Anda inginkan (misal: `D:\proyek-ai`).
3.  **Buka Anaconda Prompt:** Buka terminal Anaconda Prompt dari Start Menu.
4.  **Buat Environment Baru:** Buat "ruangan" khusus untuk proyek ini agar library tidak tercampur.
    ```bash
    conda create --name project_lubang python=3.9
    ```
5.  **Aktifkan Environment:**
    ```bash
    conda activate project_lubang
    ```
6.  **Install Library:** Instal semua library yang dibutuhkan.
    ```bash
    pip install ultralytics opencv-python
    ```

---

## 🏃 Cara Menjalankan Program

Setelah instalasi selesai, Anda dapat menjalankan program dari Anaconda Prompt (pastikan environment `(project_lubang)` sudah aktif).

1.  Pindah ke direktori (folder) proyek Anda.

    ```bash
    # Ganti D: jika perlu
    D:
    # Ganti dengan path folder Anda
    cd D:\projectpribadi\jalan-rusak-main
    ```

2.  Jalankan program dengan format:

    ```
    python main.py [NAMA_MODEL.pt] [NAMA_VIDEO.mp4]
    ```

    **Contoh Penggunaan:**

    - **Menjalankan Video 2 (dengan model `best.pt`):**
      ```bash
      python main.py best.pt video2.mp4
      ```
    - **Menjalankan Video 3 (dengan model `best_video3.pt`):**
      ```bash
      python main.py best_video3.pt video3.mp4
      ```
    - **Menjalankan Video 4 (dengan model `best_video4.pt`):**
      ```bash
      python main.py best_video4.pt video4.mp4
      ```

3.  Tekan tombol **'q'** di jendela video untuk berhenti dan keluar dari program.

---

## 📄 Lisensi

Proyek ini didistribusikan di bawah **Lisensi MIT**. Lihat file `LICENSE` untuk detail lebih lanjut.

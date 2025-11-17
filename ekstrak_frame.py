import cv2
import os

# --- KONFIGURASI ---
video_path = 'video4.mp4'
output_folder = 'gambar_baru_dari_video4'
frame_skip = 15 
# -----------------

# Membuat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Membuka video
cap = cv2.VideoCapture(video_path)
count = 0
saved_count = 0

print(f"Memulai ekstraksi frame dari {video_path}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break # Video selesai

    # Cek apakah frame ini harus disimpan
    if count % frame_skip == 0:
        # Simpan frame sebagai file gambar
        image_name = f"frame_{saved_count:04d}.jpg"
        save_path = os.path.join(output_folder, image_name)
        cv2.imwrite(save_path, frame)
        saved_count += 1
        print(f"Menyimpan {image_name}")

    count += 1

cap.release()
print(f"\nSelesai! {saved_count} gambar berhasil disimpan di folder '{output_folder}'.")
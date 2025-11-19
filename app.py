import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import subprocess

# Judul
st.title("🚗 Pothole Detection System (Final)")
st.write("Upload video, tunggu proses, dan lihat hasil yang mulus.")

# Konfigurasi Sidebar
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Tingkat Kepercayaan", 0.0, 1.0, 0.15, 0.05)

# Load Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Fungsi untuk konversi video agar bisa diputar di browser (H.264)
def convert_video(input_path, output_path):
    # Menggunakan FFmpeg yang sudah diinstal lewat packages.txt
    subprocess.call([
        'ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', output_path
    ])

# Upload File
uploaded_file = st.file_uploader("Pilih file video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Simpan file input sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Tombol untuk memulai
    if st.button("🎬 Mulai Proses Deteksi"):
        
        cap = cv2.VideoCapture(video_path)
        
        # Siapkan file output sementara
        output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_temp_file.name
        
        # Ambil info video asli
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Penulis Video (VideoWriter)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Progress Bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tracked_pothole_ids = set()
        frame_index = 0
        
        # Loop Processing (Tanpa Tampilan Live biar Cepat)
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                break

            # Tracking
            results = model.track(img, persist=True, imgsz=640, conf=confidence)
            
            frame_detections = []
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    tracked_pothole_ids.add(track_id)
                    x, y, x1, y1 = box
                    frame_detections.append(([int(x), int(y), int(x1), int(y1)]))

            # Gambar Hasil
            frame_detections.sort(key=lambda x: x[0])
            for index, box_coords in enumerate(frame_detections):
                pothole_number_in_frame = index + 1
                x, y, x1, y1 = box_coords
                
                cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
                
                # Teks Nomor (Outline + Isi)
                cv2.putText(img, str(pothole_number_in_frame), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(img, str(pothole_number_in_frame), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Teks Total (Outline + Isi)
            total_text = f"Total Lubang: {len(tracked_pothole_ids)}"
            cv2.putText(img, total_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
            cv2.putText(img, total_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            # Simpan frame ke video output
            out.write(img)
            
            # Update Progress Bar
            frame_index += 1
            progress = min(frame_index / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Memproses Frame {frame_index}/{total_frames}...")

        cap.release()
        out.release()
        
        status_text.text("Sedang mengonversi video agar kompatibel dengan browser...")
        
        # Konversi ke format yang ramah browser
        final_output_path = output_path.replace('.mp4', '_fixed.mp4')
        convert_video(output_path, final_output_path)

        status_text.success("Selesai! Silakan putar video di bawah.")
        progress_bar.empty()

        # Tampilkan Video Hasil Akhir
        st.video(final_output_path)
        
        # Tampilkan Statistik Akhir
        st.info(f"Jumlah Total Lubang Unik Terdeteksi: {len(tracked_pothole_ids)}")

        # Bersihkan file
        os.remove(video_path)
        os.remove(output_path)
        # os.remove(final_output_path) # Jangan hapus ini dulu agar bisa ditonton
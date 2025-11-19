import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import subprocess

# Judul
st.title("🚗 Pothole Detection System (Demo)")
st.write("Pilih video sampel yang tersedia di bawah ini untuk melihat demonstrasi deteksi.")

# --- SIDEBAR PENGATURAN ---
st.sidebar.header("⚙️ Pengaturan")
confidence = st.sidebar.slider("Tingkat Kepercayaan (Confidence)", 0.0, 1.0, 0.2, 0.05)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model: {e}. Pastikan file 'best.pt' ada di GitHub.")

# --- DAFTAR VIDEO ---
video_options = {
    "Jalan 1 (video.mp4)": "video.mp4",   
    "Jalan 2 (video2.mp4)": "video2.mp4",
    "Jalan 3 (video3.mp4)": "video3.mp4",
    "Jalan 4 (video4.mp4)": "video4.mp4"  
}

# --- PILIH VIDEO ---
selected_video_name = st.selectbox("📂 Pilih Video Sampel:", list(video_options.keys()))
selected_video_path = video_options[selected_video_name]

# Tampilkan Preview Video Asli
st.subheader("Preview Video Asli")
if os.path.exists(selected_video_path):
    st.video(selected_video_path)
else:
    st.error(f"File '{selected_video_path}' tidak ditemukan di GitHub! Harap upload dulu.")


# --- FUNGSI KONVERSI VIDEO ---
def convert_video(input_path, output_path):
    subprocess.call([
        'ffmpeg', '-y', '-i', input_path, '-vcodec', 'libx264', output_path
    ])

# --- TOMBOL MULAI ---
if st.button("🚀 Mulai Proses Deteksi", type="primary"):
    
    if not os.path.exists(selected_video_path):
        st.error("Video tidak ditemukan. Pastikan file video sudah di-push ke GitHub.")
    else:
        cap = cv2.VideoCapture(selected_video_path)
        
        # Siapkan output
        output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        output_path = output_temp_file.name
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # UI Progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        tracked_pothole_ids = set()
        frame_index = 0
        
        # Loop Proses
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

            # Gambar
            frame_detections.sort(key=lambda x: x[0])
            for index, box_coords in enumerate(frame_detections):
                pothole_number_in_frame = index + 1
                x, y, x1, y1 = box_coords
                
                cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
                cv2.putText(img, str(pothole_number_in_frame), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(img, str(pothole_number_in_frame), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Teks Total
            total_text = f"Total Lubang: {len(tracked_pothole_ids)}"
            cv2.putText(img, total_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
            cv2.putText(img, total_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            out.write(img)
            
            frame_index += 1
            if total_frames > 0:
                progress = min(frame_index / total_frames, 1.0)
                progress_bar.progress(progress)
            
        cap.release()
        out.release()
        
        status_text.text("Sedang finalisasi video...")
        
        # Konversi FFmpeg
        final_output_path = output_path.replace('.mp4', '_fixed.mp4')
        convert_video(output_path, final_output_path)

        status_text.success("Selesai!")
        progress_bar.empty()

        # Tampilkan Hasil
        st.subheader("Hasil Deteksi")
        st.video(final_output_path)
        
        st.info(f"Jumlah Total Lubang Unik Terdeteksi: {len(tracked_pothole_ids)}")

        # Bersihkan
        if os.path.exists(output_path): os.remove(output_path)
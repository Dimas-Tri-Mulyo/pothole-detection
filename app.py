import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os

# Judul Halaman
st.title("🚗 Pothole Detection System (YOLOv8)")
st.write("Upload video jalan raya untuk mendeteksi lubang secara otomatis.")

# Sidebar untuk konfigurasi
st.sidebar.title("Konfigurasi")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15, 0.05)

# Load Model (Cache agar tidak load berulang kali)
@st.cache_resource
def load_model():
    return YOLO("best.pt") # Pastikan file best.pt ada di GitHub kamu

model = load_model()

# Upload File Video
uploaded_file = st.file_uploader("Pilih file video...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Simpan file sementara karena OpenCV butuh path file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    st_frame = st.empty() # Placeholder untuk video
    st_text = st.empty()  # Placeholder untuk teks jumlah

    tracked_pothole_ids = set()
    frame_count = 0
    FRAME_SKIP = 3

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        # Tracking dengan YOLO
        results = model.track(img, persist=True, imgsz=640, conf=confidence)
        
        frame_detections = []
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                tracked_pothole_ids.add(track_id)
                x, y, x1, y1 = box
                frame_detections.append(([int(x), int(y), int(x1), int(y1)]))

        frame_detections.sort(key=lambda x: x[0])

        for index, box_coords in enumerate(frame_detections):
            pothole_number_in_frame = index + 1
            x, y, x1, y1 = box_coords
            
            # Gambar kotak & teks
            cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
            text_num = str(pothole_number_in_frame)
            cv2.putText(img, text_num, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, text_num, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Update teks jumlah total
        st_text.markdown(f"### Total Lubang Terdeteksi: **{len(tracked_pothole_ids)}**")
        
        # Konversi warna BGR (OpenCV) ke RGB (Browser)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Tampilkan di Web
        st_frame.image(img, channels="RGB", use_column_width=True)

    cap.release()
    os.remove(video_path) # Hapus file sementara
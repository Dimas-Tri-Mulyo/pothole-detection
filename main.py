from ultralytics import YOLO
import cv2
import numpy as np
import sys 

# --- KONFIGURASI ---
FRAME_SKIP = 3
frame_count = 0 

# Kode untuk mengambil nama file video dan model dari terminal
if len(sys.argv) < 3:
    print("---------------------------------------------------------")
    print("ERROR: Anda lupa memasukkan NAMA MODEL dan NAMA VIDEO!")
    print("CONTOH CARA RUN:")
    print("   python main.py best.pt video2.mp4")
    print("---------------------------------------------------------")
    sys.exit() 

model_path = sys.argv[1]
video_path = sys.argv[2]
# --------------------

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path) 
print(f"--- Memproses video: {video_path} dengan model: {model_path} ---")

# <-- DIKEMBALIKAN: Database untuk menghitung total lubang unik
tracked_pothole_ids = set()

while True:
    ret, img = cap.read()
    if not ret:
        break # Video selesai

    frame_count += 1 
    if frame_count % FRAME_SKIP != 0: 
        continue

    results = model.track(img, persist=True, imgsz=640, conf=0.5) 
    
    frame_detections = []
    
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            # <-- DIKEMBALIKAN: Baris ini menambahkan ID unik ke database
            tracked_pothole_ids.add(track_id)
            
            x, y, x1, y1 = box
            frame_detections.append( ([int(x), int(y), int(x1), int(y1)]) )

    # Urutkan deteksi dari kiri ke kanan
    frame_detections.sort(key=lambda x: x[0])

    # Loop untuk menggambar deteksi yang sudah diurutkan
    for index, box_coords in enumerate(frame_detections):
        pothole_number_in_frame = index + 1
        x, y, x1, y1 = box_coords
        
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
        
        # Gambar nomor urut dengan stroke (putih dengan outline hitam)
        text_num = str(pothole_number_in_frame)
        cv2.putText(img, text_num, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4) # Outline
        cv2.putText(img, text_num, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) # Isi

    # <-- DIUBAH: Teks sekarang mengambil dari hitungan total 'tracked_pothole_ids'
    total_text = f"Total Lubang Terdeteksi: {len(tracked_pothole_ids)}"
    
    # Gambar teks total dengan stroke (putih dengan outline hitam)
    cv2.putText(img, total_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5) # Outline
    cv2.putText(img, total_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) # Isi
            
    # Perbaikan untuk video "gepeng" (menjaga rasio aspek)
    display_h = 720 
    
    if 'img' in locals() and img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        h_orig, w_orig, _ = img.shape
        ratio = display_h / h_orig
        display_w = int(w_orig * ratio)
        display_img = cv2.resize(img, (display_w, display_h))
        cv2.imshow('Deteksi Jalan Berlubang', display_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Video Selesai ---
cap.release()
cv2.destroyAllWindows()

# Tampilkan juga total akhir di terminal untuk konfirmasi
print(f"--- Selesai ---")
print(f"Total lubang unik yang terdeteksi selama video: {len(tracked_pothole_ids)}")
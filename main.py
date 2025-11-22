import os
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "input", "input_video.mp4")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n-face.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "people")
os.makedirs(OUTPUT_DIR, exist_ok=True)

face_detector = YOLO(YOLO_MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Cannot open video:", VIDEO_PATH)
    exit()

known_faces = []  
person_id = 0
SIMILARITY_THRESHOLD = 0.6  

def compare_faces(face1, face2):
    """Compare two faces using histogram similarity"""
    face1_gray = cv2.cvtColor(face1, cv2.COLOR_BGR2GRAY)
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    h1 = cv2.calcHist([face1_gray], [0], None, [256], [0,256])
    h2 = cv2.calcHist([face2_gray], [0], None, [256], [0,256])
    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)
    sim = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return sim

# Process Video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = face_detector(frame, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Compare with known faces
            matched = False
            for idx, known in enumerate(known_faces):
                sim = compare_faces(face_crop, known)
                if sim > SIMILARITY_THRESHOLD:
                    folder = os.path.join(OUTPUT_DIR, f"person_{idx}")
                    os.makedirs(folder, exist_ok=True)
                    count = len(os.listdir(folder))
                    cv2.imwrite(os.path.join(folder, f"face_{count}.jpg"), face_crop)
                    matched = True
                    break

            # New person
            if not matched:
                known_faces.append(face_crop)
                folder = os.path.join(OUTPUT_DIR, f"person_{person_id}")
                os.makedirs(folder, exist_ok=True)
                cv2.imwrite(os.path.join(folder, "face_0.jpg"), face_crop)
                person_id += 1

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    # Show video
    cv2.imshow("YOLO Face Grouping", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

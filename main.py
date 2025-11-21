import cv2
import os
import urllib.request

# Model directory
model_dir = "face_model"
os.makedirs(model_dir, exist_ok=True)

prototxt_path = os.path.join(model_dir, "deploy.prototxt")
weights_path  = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# Download deploy.prototxt if missing
if not os.path.isfile(prototxt_path):
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        prototxt_path
    )

# Download weights if missing
if not os.path.isfile(weights_path):
    urllib.request.urlretrieve(
        "https://huggingface.co/DebajyotyBanik/Enhancing-Security-with-Optimized-Masked-Face-Recognition-and-Mask-Detection/resolve/main/res10_300x300_ssd_iter_140000.caffemodel",
        weights_path
    )

net = cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

# VIDEO INPUT (CHANGE THIS)
video_path = r"C:/Users/Eimdadul Haque/Desktop/OneOrder/input/input_video.mp4"

if not os.path.isfile(video_path):
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    exit()


# SAVE FACES
save_folder = "faces"
os.makedirs(save_folder, exist_ok=True)
face_id = 1

# PROCESS VIDEO FRAMES
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Prepare frame for DNN
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    detections = net.forward()

    # Loop detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:  # threshold
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            x1, y1, x2, y2 = box.astype("int")

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Crop and save face
            face = frame[y1:y2, x1:x2]
            if face.size > 0:
                save_path = f"{save_folder}/face_{face_id}.jpg"
                cv2.imwrite(save_path, face)
                face_id += 1

    cv2.imshow("Face Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


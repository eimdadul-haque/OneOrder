import cv2

# Load the video
cap = cv2.VideoCapture("input/input_video.mp4")

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter to save the output
out = cv2.VideoWriter("output/output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

x1, y1 = 400, 300 
x2, y2 = 1100, 600

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a rectangle (BGR color format, thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    # Write the frame to output
    out.write(frame)

    # Optional: display the video while processing
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

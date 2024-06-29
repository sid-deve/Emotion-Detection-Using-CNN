import cv2
from fer import FER
import time
from datetime import datetime

# Try different backends if needed
backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]
cap = None

print("Starting webcam initialization...")

for backend in backends:
    print(f"Trying backend: {backend}")
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"Webcam opened successfully with backend {backend}.")
        break
    else:
        print(f"Failed to open webcam with backend {backend}.")

if not cap or not cap.isOpened():
    print("Error: Could not open webcam with any backend.")
    exit()

# Optional: Add a delay to ensure the camera is ready
print("Waiting for the camera to initialize...")
time.sleep(2)

print("Testing frame capture in isolation...")
ret, frame = cap.read()
if not ret:
    print("Error: Failed to grab a frame.")
    cap.release()
    exit()
else:
    print("Successfully grabbed a frame.")

# Check frame size and type for debugging
print(f"Frame type: {type(frame)}, Frame shape: {frame.shape if frame is not None else 'None'}")

detector = FER()

print("Starting the emotion detection loop...")

# Set the duration for displaying the output before pausing (in seconds)
display_duration = 10
pause_duration = 5
start_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    print(f"Captured frame: {frame.shape}")

    emotions = detector.detect_emotions(frame)
    print(f"Detected emotions: {emotions}")
    
    for emotion in emotions:
        (x, y, w, h) = emotion["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        for emotion_name, score in emotion["emotions"].items():
            text = f"{emotion_name}: {score:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y -= 20

    cv2.imshow('Emotion Detection', frame)

    # Check if the display duration has been exceeded
    elapsed_time = time.time() - start_time
    if elapsed_time > display_duration:
        print("Specified display duration reached. Pausing output.")
        # Keep displaying the last frame for the pause duration
        pause_start_time = time.time()
        while time.time() - pause_start_time < pause_duration:
            cv2.imshow('Emotion Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting loop and releasing resources.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        # Reset the start time after the pause
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting loop and releasing resources.")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam and windows closed.")

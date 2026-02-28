import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera")
    print("Make sure to grant camera permissions in System Preferences > Security & Privacy > Camera")
    exit()

print("Camera opened successfully! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip frame for selfie view
    frame = cv2.flip(frame, 1)
    
    # Add text overlay
    cv2.putText(frame, "Gesture Detector", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'q' to quit", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    cv2.imshow("Gesture Detector", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Program closed")
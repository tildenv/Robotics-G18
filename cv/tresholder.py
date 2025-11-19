import cv2
import numpy as np

# A function that does nothing, needed for the trackbar
def nothing(x):
    pass

# Set up the webcam
video_path = "/home/stas/Videos/Screencasts/Screencast from 2025-11-11 15-28-43.webm"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create a window to hold the controls
cv2.namedWindow("Controls")
cv2.resizeWindow("Controls", 500, 300)

# Create trackbars for HSV (Hue, Saturation, Value)
# Hue: 0-179 (in OpenCV)
cv2.createTrackbar("H_min", "Controls", 0, 179, nothing)
cv2.createTrackbar("H_max", "Controls", 179, 179, nothing)
# Saturation: 0-255
cv2.createTrackbar("S_min", "Controls", 0, 255, nothing)
cv2.createTrackbar("S_max", "Controls", 255, 255, nothing)
# Value: 0-255
cv2.createTrackbar("V_min", "Controls", 0, 255, nothing)
cv2.createTrackbar("V_max", "Controls", 255, 255, nothing)

print("Adjust the sliders in the 'Controls' window.")
print("Press 'q' to quit.")

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the current positions of the trackbars
    h_min = cv2.getTrackbarPos("H_min", "Controls")
    h_max = cv2.getTrackbarPos("H_max", "Controls")
    s_min = cv2.getTrackbarPos("S_min", "Controls")
    s_max = cv2.getTrackbarPos("S_max", "Controls")
    v_min = cv2.getTrackbarPos("V_min", "Controls")
    v_max = cv2.getTrackbarPos("V_max", "Controls")

    # Create the lower and upper bounds as numpy arrays
    lower_bound = np.array([h_min, s_min, v_min])
    upper_bound = np.array([h_max, s_max, v_max])

    # Create the mask using the inRange function
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # --- Optional: Clean up the mask ---
    # Erode to remove small noise specks
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.erode(mask, kernel, iterations=1)
    # Dilate to fill in holes in the main object
    mask_cleaned = cv2.dilate(mask_cleaned, kernel, iterations=1)
    # --- End of cleanup ---

    # Display the original frame
    cv2.imshow("Original Frame", frame)
    # Display the resulting mask
    cv2.imshow("Mask", mask_cleaned)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(3000) & 0xFF == ord('q'):
        break

# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

# --- Define your color ranges here ---
# This is a dictionary where each color has its lower and upper HSV bounds
# You MUST replace these with the values you found using Script 1!
COLOR_RANGES = {
    "red": {
        "lower": np.array([0, 150, 100]),
        "upper": np.array([10, 255, 255])
        # Note: Red can be tricky as it wraps around 0/179.
        # You might need two ranges for red.
    },
    "blue": {
        "lower": np.array([100, 150, 50]),
        "upper": np.array([130, 255, 255])
    },
    "green": {
        "lower": np.array([40, 100, 50]),
        "upper": np.array([80, 255, 255])
    }
    # ... add more colors like "yellow", "orange", etc.
}

# --- Morphology Kernel ---
# We define this once to clean up all masks
kernel = np.ones((5, 5), np.uint8)


def find_color_smarties(hsv_image, frame, color_name, bounds):
    """Finds and draws contours for a specific color."""
    
    # 1. Create the mask
    mask = cv2.inRange(hsv_image, bounds["lower"], bounds["upper"])
    
    # 2. Clean up the mask
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # 3. Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Filter and draw
    for cnt in contours:
        # Filter by area to remove small noise
        area = cv2.contourArea(cnt)
        if area > 500:  # You may need to tune this '500' value
            
            # Get a bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw the rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Put the color name text
            cv2.putText(frame, color_name, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# --- Main Program Loop ---

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Loop through all our defined colors and find them
    for color_name, bounds in COLOR_RANGES.items():
        find_color_smarties(hsv, frame, color_name, bounds)

    # Show the final result
    cv2.imshow("Smartie Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
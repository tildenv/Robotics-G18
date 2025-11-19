import cv2
import numpy as np

# --- SETTINGS ---
# This is the most important setting.
# It's the number of colors to find.
# Good starting point: (Number of Smartie colors) + 1 (for the background)
K = 9 

# Width to shrink the image to for K-Means processing.
# Smaller is MUCH faster. Larger is more accurate but will lag.
PROCESS_WIDTH = 200

# Minimum contour area (in the *small* image) to be considered an object.
# This filters out tiny specks of noise.
MIN_AREA = 50

# --- "IS COLOR" DEFINITION ---
# We use general HSV thresholds to decide if a K-Means cluster
# center is "red". This is more robust than BGR.
#
# --- EDITED THRESHOLDS ---
# We are making this MUCH stricter to exclude orange and brown.
# Orange has a hue > 10. Brown is desaturated or dark.
#
# HUE: Red is in two ranges (0-8 and 172-179) in OpenCV.
#      (Narrowed from 0-10 and 170-179)
# SATURATION: > 150 (Increased from 120 to exclude desaturated browns)
# VALUE: > 130 (Increased from 100 to exclude dark browns)
def is_color_red(bgr_color):
    """Checks if a BGR color is 'red' based on general HSV ranges."""
    # Convert the single BGR pixel to HSV
    # Needs to be a 3D array: [[(B, G, R)]]
    color_3d = np.uint8([[bgr_color]])
    color_hsv = cv2.cvtColor(color_3d, cv2.COLOR_BGR2HSV)[0][0]
    
    hue = int(color_hsv[0])
    saturation = int(color_hsv[1])
    value = int(color_hsv[2])
    
    # Check if Hue is in the red ranges and Sat/Val are high enough
    is_red_hue = (hue <= 8 or hue >= 172)
    is_saturated = saturation > 150
    is_bright = value > 130
    
    return is_red_hue and is_saturated and is_bright
# ---

# Set up the webcam
# video_path = "/home/stas/Videos/Screencasts/Screencast from 2025-11-11 15-28-43.webm"
video_path = "/dev/ttyACM0"
video_path = 2
# baudrate = 1000000
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

print(f"Starting automated K-Means detector for RED with K={K}...")
print(f"Processing on a {PROCESS_WIDTH}px wide image for speed.")
print("Press 'q' to quit.")

while True:
    # 1. READ AND PRE-PROCESS FRAME
    ret, frame = cap.read()
    if not ret:
        break

    # Get original dimensions
    original_height, original_width, _ = frame.shape
    
    # Calculate new height to maintain aspect ratio
    process_height = int((PROCESS_WIDTH / original_width) * original_height)
    
    # Resize to a small image for fast K-Means processing
    small_frame = cv2.resize(frame, (PROCESS_WIDTH, process_height), 
                             interpolation=cv2.INTER_AREA)

    # 2. RUN K-MEANS CLUSTERING
    # Reshape the image into a long list of pixels (N_pixels, 3_channels)
    pixel_data = small_frame.reshape((-1, 3))
    
    # Convert to float32, as required by cv2.kmeans
    pixel_data = np.float32(pixel_data)

    # Define the criteria for the algorithm to stop
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Run K-Means.
    compactness, labels, centers = cv2.kmeans(pixel_data, 
                                              K, 
                                              None, 
                                              criteria, 
                                              10, 
                                              cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers back to 8-bit values (0-255)
    centers = np.uint8(centers)

    # 3. CREATE QUANTIZED IMAGE & FIND BACKGROUND
    # Map each pixel's label back to its center color
    quantized_data = centers[labels.flatten()]
    
    # Reshape the data back into a small image
    quantized_frame = quantized_data.reshape((small_frame.shape))
    
    # Find the most frequent cluster label
    # This is assumed to be the background
    unique_labels, counts = np.unique(labels, return_counts=True)
    background_cluster_index = unique_labels[np.argmax(counts)]

    # 4. FIND CONTOURS FOR "RED" CLUSTERS
    
    # Create a copy of the original frame to draw on
    output_frame = frame.copy()

    # Calculate scaling factors to draw on the original frame
    scale_x = original_width / PROCESS_WIDTH
    scale_y = original_height / process_height

    # Loop through each of the K centers
    for i, color_center in enumerate(centers):
        
        # Check if this cluster is the background OR if it's not red
        if i == background_cluster_index or not is_color_red(color_center):
            continue
            
        # This cluster is NOT background and IS red, so let's find it.
            
        # Get the BGR color for drawing (it's already a 'red' color)
        color_bgr = (int(color_center[0]), int(color_center[1]), int(color_center[2]))
        
        # Create a mask for *only* this cluster's color
        mask = cv2.inRange(quantized_frame, color_center, color_center)
        
        # Find contours on this mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop through all contours found for this color
        for cnt in contours:
            # Get area of the contour (in the small image)
            area = cv2.contourArea(cnt)
            
            # Filter out small noise
            if area > MIN_AREA:
                
                # --- Scale contour back to original frame size ---
                cnt_scaled = (cnt * [scale_x, scale_y]).astype(int)
                
                # Get the bounding box of the scaled contour
                x, y, w, h = cv2.boundingRect(cnt_scaled)
                
                # Draw the rectangle on the *original* output frame
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                
                # Draw the label
                cv2.putText(output_frame, "Red", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 5. DISPLAY RESULTS
    
    # Resize the quantized image back up to full size for display
    quantized_display = cv2.resize(quantized_frame, (original_width, original_height), 
                                   interpolation=cv2.INTER_NEAREST)

    cv2.imshow(f"K-Means Quantized (K={K})", quantized_display)
    cv2.imshow("Automated Red Detector", output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
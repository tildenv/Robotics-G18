import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
# Path to your folder containing the images
# Example: 'C:/Users/Name/Desktop/calibration_images' or './images'
images_dir = './photos'
file_extension = '*.jpg' # Change to *.png or *.jpeg if necessary

# Chessboard dimensions (internal corners)
# Based on your previous image (5x7 squares = 4x6 internal corners)
# NOTE: Verify this matches YOUR checkerboard! Count internal corners, not squares.
# IMPORTANT: Try different combinations if detection fails: (6,4), (7,9), (8,6), etc.
CHECKERBOARD = (5, 4)  # (width, height) in corners - SWAPPED to try alternate orientation
SQUARE_SIZE = 15.0  # Square size in mm (measure your actual checkerboard)

# --- SETUP ---
# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Get list of images
image_files = glob.glob(os.path.join(images_dir, file_extension))

if not image_files:
    print(f"No images found in {images_dir} with extension {file_extension}")
    exit()

print(f"Found {len(image_files)} images. Processing...")

# --- PROCESSING LOOP ---
img_shape = None

successful_images = 0

for fname in image_files:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read: {os.path.basename(fname)}")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img_shape is None:
        img_shape = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(
        gray, 
        CHECKERBOARD, 
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret == True:
        objpoints.append(objp)

        # Refine pixel coordinates
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        
        successful_images += 1
        print(f"✓ Found corners in: {os.path.basename(fname)}")

        # Save visualization for the first successful detection
        if successful_images == 1:
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            output_path = os.path.join(images_dir, f"detected_corners_{os.path.basename(fname)}")
            cv2.imwrite(output_path, img)
            print(f"  → Saved visualization to: {output_path}")
    else:
        print(f"✗ Could not find corners in: {os.path.basename(fname)}")

print(f"\nSuccessfully processed {successful_images}/{len(image_files)} images")

# --- CALIBRATION ---
if len(objpoints) >= 3:  # Need at least 3 images for calibration
    print("\nCalibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\n--- SUCCESS ---")
    print("Camera Matrix (K):\n", mtx)
    print("\nDistortion Coefficients (D):\n", dist)

    # --- CALCULATE ERROR ---
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    print(f"\nMean Reprojection Error: {total_error / len(objpoints):.4f} pixels")

    # --- SAVE RESULT ---
    # Save the camera matrix and distortion coefficients for later use
    output_file = os.path.join(images_dir, "camera_calibration_data.npz")
    np.savez(output_file, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print(f"\nCalibration data saved to '{output_file}'")
    
    # Also save as text for easy viewing
    txt_file = os.path.join(images_dir, "camera_calibration.txt")
    with open(txt_file, 'w') as f:
        f.write("Camera Calibration Results\n")
        f.write("="*50 + "\n\n")
        f.write("Camera Matrix (K):\n")
        f.write(str(mtx) + "\n\n")
        f.write("Distortion Coefficients (D):\n")
        f.write(str(dist) + "\n\n")
        f.write(f"Mean Reprojection Error: {total_error / len(objpoints):.4f} pixels\n")
    print(f"Human-readable results saved to '{txt_file}'")

else:
    print(f"\nError: Need at least 3 images with detected corners for calibration.")
    print(f"Only found corners in {successful_images} image(s).")
    print("Tips:")
    print("  - Verify CHECKERBOARD dimensions match your pattern")
    print("  - Ensure good lighting and sharp focus")
    print("  - Try different angles and distances")
#!/usr/bin/env python3
"""
Object Detection with Camera Calibration
Detects red objects using blob detection and calculates their 2D world coordinates
using camera calibration data, assuming objects are at Z=0 (table plane).
"""

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from skimage import color, feature
import time
import os


class ObjectDetector:
    """
    Detector for red objects with calibrated 2D position estimation.
    """
    
    def __init__(self, calibration_file='./photos/camera_calibration_data.npz', camera_height_mm=None):
        """
        Initialize the object detector.
        
        Args:
            calibration_file: Path to the camera calibration .npz file
            camera_height_mm: Height of camera above the table (Z=0 plane) in mm.
                            If None, will be estimated from robot FK or manual input.
        """
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_height = camera_height_mm
        
        # Load calibration data
        self.load_calibration(calibration_file)
        
    def load_calibration(self, calibration_file):
        """Load camera calibration data from file."""
        if not os.path.exists(calibration_file):
            print(f"Warning: Calibration file not found: {calibration_file}")
            print("Detection will work but 3D position calculation will not be available.")
            return False
        
        try:
            calib_data = np.load(calibration_file)
            self.camera_matrix = calib_data['mtx']
            self.dist_coeffs = calib_data['dist']
            print("Camera calibration loaded successfully!")
            print(f"Camera Matrix:\n{self.camera_matrix}")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def process_hsv_image(self, img):
        """
        Simple processing of a color (HSV) image.
        Create a black and white picture where the red objects are white.
        
        Args:
            img: HSV image (normalized 0-1 from skimage)
            
        Returns:
            Binary mask with red objects as white
        """
        hue = img[:, :, 0]
        sat = img[:, :, 1]
        val = img[:, :, 2]

        # Red is near 0 or 1 in hue space (wraps around)
        mask = ((hue < 0.05) | (hue > 0.95)) & (sat > 0.35) & (val > 0.2)
        mask = img_as_ubyte(mask)

        # Remove noise
        mask = cv2.medianBlur(mask, 5)  # median filter
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def process_rgb_image(self, img):
        """
        Segmentation of red structures in the RGB channel.
        
        Args:
            img: RGB image (0-255 uint8)
            
        Returns:
            Binary mask with red objects as white
        """
        r_comp = img[:, :, 0]
        g_comp = img[:, :, 1]
        b_comp = img[:, :, 2]
        
        segm = (r_comp > 160) & (r_comp < 180) & (g_comp > 50) & (g_comp < 80) & \
               (b_comp > 50) & (b_comp < 80)
        
        return img_as_ubyte(segm)
    
    def blob_detection(self, mask, display_frame=None, min_sigma=20, max_sigma=40, threshold=0.2):
        """
        Detect blobs in binary mask using Difference of Gaussian (DoG).
        
        Args:
            mask: Binary mask image
            display_frame: Optional frame to draw detections on (BGR format)
            min_sigma: Minimum sigma for DoG
            max_sigma: Maximum sigma for DoG
            threshold: Detection threshold
            
        Returns:
            Array of detected blobs (y, x, radius)
        """
        blobs = feature.blob_dog(mask, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
        blobs[:, 2] = blobs[:, 2] * (2 ** 0.5)
        
        if display_frame is not None and blobs.size > 0:
            for i, (y, x, r) in enumerate(blobs, start=1):
                if 5 <= r <= 30:  # Filter by reasonable radius
                    cv2.circle(display_frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                    cv2.circle(display_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    cv2.putText(display_frame, f"#{i} ({int(x)},{int(y)}) r={int(r)}",
                              (int(x)+6, int(y)-6), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.45, (0, 255, 0), 1, cv2.LINE_AA)
        
        return blobs
    
    def pixel_to_world_coords(self, pixel_x, pixel_y, Z_world=0.0):
        """
        Convert pixel coordinates to 2D camera frame coordinates (X, Y) at given Z depth.
        
        This uses the pinhole camera model to back-project from image plane:
        [u]   [fx  0  cx]   [X]
        [v] = [0  fy cy] * [Y] / Z
        [1]   [0   0  1]   [Z]
        
        Solving for X and Y at known Z:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        NOTE: This returns coordinates in the CAMERA FRAME, where:
        - X is right (in image)
        - Y is down (in image)
        - Z is depth (distance from camera, perpendicular to image plane)
        
        For a camera looking down at a table:
        - Table is at Z = camera_height in camera frame
        - Objects on table have Z coordinate = camera_height
        
        Args:
            pixel_x: X coordinate in pixels (u)
            pixel_y: Y coordinate in pixels (v)
            Z_world: Z coordinate in world/base frame (not used, kept for compatibility)
            
        Returns:
            (X_camera, Y_camera) in mm at Z=camera_height, or None if calibration not loaded
        """
        if self.camera_matrix is None:
            print("Warning: Camera calibration not loaded")
            return None
        
        if self.camera_height is None:
            print("Warning: Camera height not set. Cannot calculate coordinates.")
            return None
        
        # Extract camera parameters
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # For objects on the table, the Z distance in camera frame equals camera height
        # (assuming camera is looking down at the table)
        Z_camera = self.camera_height
        
        if Z_camera <= 0:
            print("Warning: Invalid camera height")
            return None
        
        # Back-project from image plane to 3D camera frame coordinates
        # These are in the camera's coordinate system where Z is the depth
        X_camera = (pixel_x - cx) * Z_camera / fx
        Y_camera = (pixel_y - cy) * Z_camera / fy
        
        return (X_camera, Y_camera)
    
    def detect_objects_with_positions(self, frame, use_hsv=True):
        """
        Detect objects and calculate their world positions.
        
        Args:
            frame: Input BGR image from camera
            use_hsv: If True, use HSV segmentation. If False, use RGB.
            
        Returns:
            List of detections: [(pixel_x, pixel_y, radius, world_x, world_y), ...]
        """
        # Convert to RGB
        image_rgb = frame[:, :, ::-1]
        
        # Segment red objects
        if use_hsv:
            image_hsv = color.rgb2hsv(image_rgb)
            mask = self.process_hsv_image(image_hsv)
        else:
            mask = self.process_rgb_image(image_rgb)
        
        # Detect blobs
        blobs = self.blob_detection(mask)
        
        # Calculate world positions
        detections = []
        for y, x, r in blobs:
            if 5 <= r <= 30:  # Filter by reasonable radius
                world_coords = self.pixel_to_world_coords(x, y)
                detections.append((x, y, r, world_coords))
        
        return detections, mask
    
    def set_camera_height(self, height_mm):
        """Set the camera height above the table plane (Z=0)."""
        self.camera_height = height_mm
        # print(f"Camera height set to {height_mm} mm")
    
    def show_in_moved_window(self, win_name, img, x, y):
        """Show an image in a window at specified position."""
        cv2.namedWindow(win_name)
        cv2.moveWindow(win_name, x, y)
        cv2.imshow(win_name, img)
    
    def run_camera_detection(self, camera_id=2, use_hsv=True):
        """
        Run real-time object detection with camera feed.
        
        Args:
            camera_id: Camera device ID or URL
            use_hsv: If True, use HSV segmentation. If False, use RGB.
        """
        print("Opening camera...")
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Cannot open camera")
            raise RuntimeError("Cannot open camera")
        
        print("Starting camera loop (Press 'q' to quit, 'h' to toggle HSV/RGB, 's' to save)")
        
        # FPS tracking
        old_time = time.perf_counter()
        fps = 0
        stop = False
        
        while not stop:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break
            
            # Detect objects
            detections, mask = self.detect_objects_with_positions(frame, use_hsv=use_hsv)
            
            # Draw detections on frame
            for i, (x, y, r, world_coords) in enumerate(detections, start=1):
                cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                
                # Display pixel coordinates
                text = f"#{i} ({int(x)},{int(y)}) r={int(r)}"
                cv2.putText(frame, text, (int(x)+6, int(y)-6), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                
                # Display world coordinates if available
                if world_coords is not None:
                    wx, wy = world_coords
                    text_world = f"World: ({wx:.1f}, {wy:.1f}) mm"
                    cv2.putText(frame, text_world, (int(x)+6, int(y)+10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Update FPS
            new_time = time.perf_counter()
            time_dif = new_time - old_time
            old_time = new_time
            fps = fps * 0.95 + 0.05 * 1 / time_dif
            
            # Display info
            mode_str = "HSV" if use_hsv else "RGB"
            str_out = f"FPS: {int(fps)} | Mode: {mode_str} | Objects: {len(detections)}"
            cv2.putText(frame, str_out, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            if self.camera_height:
                height_str = f"Camera Height: {self.camera_height} mm"
                cv2.putText(frame, height_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                warning_str = "Warning: Camera height not set!"
                cv2.putText(frame, warning_str, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                          0.7, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Display windows
            self.show_in_moved_window('Detection', frame, 0, 10)
            self.show_in_moved_window('Mask', mask, 700, 10)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop = True
            elif key == ord('h'):
                use_hsv = not use_hsv
                print(f"Switched to {'HSV' if use_hsv else 'RGB'} mode")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"detection_{timestamp}.jpg", frame)
                print(f"Saved detection_{timestamp}.jpg")
        
        print("Stopping detection loop")
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point for testing."""
    # Create detector
    detector = ObjectDetector(calibration_file='./photos/camera_calibration_data.npz')
    
    # Set camera height (you need to measure or get from robot FK)
    # Example: If camera is mounted 250mm above the table
    print("\nEnter camera height above table (Z=0 plane) in mm:")
    print("(You can measure this or get it from robot forward kinematics)")
    try:
        height = float(input("Camera height (mm): "))
        detector.set_camera_height(height)
    except ValueError:
        print("Invalid input. Proceeding without camera height (position calculation disabled)")
    
    # Run detection
    try:
        detector.run_camera_detection(camera_id=2, use_hsv=True)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

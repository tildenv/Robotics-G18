#!/usr/bin/env python3
"""
Diagnostic tool to find the correct checkerboard pattern dimensions.
Tries multiple common checkerboard sizes and reports which ones work.
"""
import cv2
import numpy as np
import glob
import os

# --- CONFIGURATION ---
images_dir = './photos'
file_extension = '*.jpg'

# Common checkerboard patterns to try (width, height) in internal corners
PATTERNS_TO_TRY = [
    (9, 6), (6, 9),  # 10x7 squares
    (8, 6), (6, 8),  # 9x7 squares
    (7, 5), (5, 7),  # 8x6 squares
    (9, 7), (7, 9),  # 10x8 squares
    (8, 5), (5, 8),  # 9x6 squares
    (6, 4), (4, 6),  # 7x5 squares (what we tried)
    (7, 4), (4, 7),  # 8x5 squares
    (10, 7), (7, 10),  # 11x8 squares
]

# Get first image to test
image_files = glob.glob(os.path.join(images_dir, file_extension))

if not image_files:
    print(f"No images found in {images_dir}")
    exit()

test_image = image_files[0]
print(f"Testing image: {os.path.basename(test_image)}")
print(f"Trying {len(PATTERNS_TO_TRY)} different checkerboard patterns...\n")

img = cv2.imread(test_image)
if img is None:
    print(f"Could not read image: {test_image}")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels\n")

found_patterns = []

for pattern in PATTERNS_TO_TRY:
    ret, corners = cv2.findChessboardCorners(
        gray,
        pattern,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        found_patterns.append(pattern)
        print(f"✓ FOUND: {pattern[0]}x{pattern[1]} internal corners ({pattern[0]+1}x{pattern[1]+1} squares)")

if found_patterns:
    print(f"\n{'='*60}")
    print(f"SUCCESS! Found {len(found_patterns)} matching pattern(s):")
    for pattern in found_patterns:
        print(f"  CHECKERBOARD = {pattern}")
    print(f"{'='*60}")
    print(f"\nUpdate calibration.py with one of the patterns above.")
    
    # Test on all images with the first found pattern
    print(f"\nTesting pattern {found_patterns[0]} on all {len(image_files)} images:")
    success_count = 0
    for fname in image_files:
        img = cv2.imread(fname)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.findChessboardCorners(
            gray,
            found_patterns[0],
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if ret:
            success_count += 1
            print(f"  ✓ {os.path.basename(fname)}")
        else:
            print(f"  ✗ {os.path.basename(fname)}")
    
    print(f"\n{success_count}/{len(image_files)} images detected successfully")
else:
    print(f"\n{'='*60}")
    print("❌ NO PATTERNS FOUND!")
    print("{'='*60}")
    print("\nPossible issues:")
    print("  1. Images don't contain a standard checkerboard pattern")
    print("  2. Checkerboard is too blurry or poorly lit")
    print("  3. Checkerboard has an unusual size not in common patterns")
    print("  4. Images are corrupted or wrong format")
    print("\nManual check:")
    print(f"  - Open: {test_image}")
    print("  - Count internal corners (where 4 black squares meet)")
    print("  - Horizontally: ___")
    print("  - Vertically: ___")
    print("  - Then add pattern manually to PATTERNS_TO_TRY")

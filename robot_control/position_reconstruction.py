#!/usr/bin/env python3
"""
Position Reconstruction Module
Handles reconstruction of 3D object positions from camera detections.
Contains coordinate transformation logic from pixel coordinates to robot base frame.
"""

import numpy as np
from typing import Tuple, Optional


class PositionReconstructor:
    """
    Reconstructs 3D positions of detected objects in robot base frame.
    
    This class handles the complete transformation pipeline:
    1. Pixel coordinates -> Camera frame (via pinhole model in ObjectDetector)
    2. Camera frame -> Base frame (via robot kinematics)
    3. Position refinement for table-mounted objects
    """
    
    def __init__(self, robot_controller):
        """
        Initialize the position reconstructor.
        
        Args:
            robot_controller: RobotController instance for FK/IK and transformations
        """
        self.robot = robot_controller
        
    def reconstruct_position(
        self, 
        detection: Tuple[float, float, float, Optional[Tuple[float, float]]],
        hover_height: float = 20.0,
        table_z: float = 0.0
    ) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Reconstruct 3D position in base frame from a detection.
        
        Args:
            detection: Tuple of (x_pixel, y_pixel, radius, world_coords)
                where world_coords is (wx, wy) in camera frame from pinhole model
            hover_height: Height in mm to hover above the detected object (default: 20mm)
            table_z: Z-coordinate of table surface in base frame (default: 0mm)
            
        Returns:
            Tuple of (target_position, debug_info) or None if reconstruction fails:
                - target_position: np.ndarray [x, y, z] in base frame (mm)
                - debug_info: dict with intermediate values for debugging
        """
        x_pixel, y_pixel, radius, world_coords = detection
        
        if world_coords is None:
            return None
        
        wx, wy = world_coords
        
        # Step 1: Get current camera position and orientation from robot FK
        camera_pos, camera_rot = self.robot.get_camera_position()
        if camera_pos is None:
            return None
        
        camera_height = camera_pos[2]  # Z coordinate of camera in base frame
        
        # Step 2: Build 3D point in camera frame
        # Due to the camera's physical mounting orientation, we need to swap coordinates
        # The rotation matrix shows the camera frame is rotated relative to the stylus frame
        # Empirically determined mapping: [camera_height, -wy, wx]
        camera_coords = np.array([camera_height, -wy, wx])
        
        # Step 3: Transform from camera frame to base frame using the kinematic chain
        base_coords = self.robot.camera_to_base_frame(camera_coords)
        if base_coords is None:
            return None
        
        # Apply empirical correction
        base_coords[1] = -1 * base_coords[1]
        
        # Step 4: Adjust Z coordinate for table-mounted objects
        # The object should be on the table surface, so we override Z
        # and add the hover height
        target_z = table_z + hover_height
        base_coords[2] = target_z
        
        # Prepare debug information
        debug_info = {
            'pixel_coords': (x_pixel, y_pixel),
            'camera_frame_from_pinhole': (wx, wy),
            'camera_position': camera_pos,
            'camera_rotation': camera_rot,
            'camera_height': camera_height,
            'camera_coords_3d': camera_coords.copy(),
            'base_coords_raw': base_coords.copy(),
            'table_z': table_z,
            'hover_height': hover_height,
            'target_coords': base_coords.copy()
        }
        
        return base_coords, debug_info
    
    def format_debug_info(self, debug_info: dict, detection_number: int) -> str:
        """
        Format debug information as a human-readable string.
        
        Args:
            debug_info: Debug information dictionary from reconstruct_position
            detection_number: Detection index (1-based) for display
            
        Returns:
            Formatted string with debug information
        """
        x_pixel, y_pixel = debug_info['pixel_coords']
        wx, wy = debug_info['camera_frame_from_pinhole']
        cam_pos = debug_info['camera_position']
        cam_rot = debug_info['camera_rotation']
        cam_height = debug_info['camera_height']
        cam_coords = debug_info['camera_coords_3d']
        base_raw = debug_info['base_coords_raw']
        target = debug_info['target_coords']
        
        output = []
        output.append(f"Detection #{detection_number}:")
        output.append(f"  Pixel coords: ({int(x_pixel)}, {int(y_pixel)})")
        output.append(f"  Camera frame coords from pinhole: ({wx:.1f}, {wy:.1f}) mm")
        output.append(f"  Camera position in base: ({cam_pos[0]:.1f}, {cam_pos[1]:.1f}, {cam_pos[2]:.1f}) mm")
        output.append(f"  Camera rotation matrix:\n{cam_rot}")
        output.append(f"  Object in camera frame: ({cam_coords[0]:.1f}, {cam_coords[1]:.1f}, {cam_coords[2]:.1f}) mm")
        output.append(f"  Object in base frame (raw): ({base_raw[0]:.1f}, {base_raw[1]:.1f}, {base_raw[2]:.1f}) mm")
        output.append(f"  Target in base frame (hover): ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) mm")
        
        return "\n".join(output)
    
    def reconstruct_all_detections(
        self, 
        detections: list,
        hover_height: float = 20.0,
        table_z: float = 0.0
    ) -> list:
        """
        Reconstruct positions for all detections.
        
        Args:
            detections: List of detection tuples
            hover_height: Height in mm to hover above objects (default: 20mm)
            table_z: Z-coordinate of table surface (default: 0mm)
            
        Returns:
            List of tuples (target_position, debug_info) for successful reconstructions
            Empty list entries for failed reconstructions
        """
        results = []
        for detection in detections:
            result = self.reconstruct_position(detection, hover_height, table_z)
            results.append(result)
        return results

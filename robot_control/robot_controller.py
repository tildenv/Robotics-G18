#!/usr/bin/env python3
"""
Robot Controller
A simple class for controlling Dynamixel-based robot arms.
Provides easy initialization and joint control functionality.
"""

import dynamixel_sdk as dxl
import time
import math
import numpy as np
from typing import List, Optional, Tuple


class RobotController:
    """
    A controller class for managing Dynamixel servo motors in a robot arm.
    
    Attributes:
        port_name (str): Serial port name (e.g., '/dev/ttyACM0')
        baudrate (int): Communication baudrate
        motor_ids (List[int]): List of motor IDs
        protocol_version (float): Dynamixel protocol version (1.0 or 2.0)
    """
    
    # Control table addresses for MX/AX series (Protocol 1.0)
    ADDR_TORQUE_ENABLE = 24
    ADDR_CW_COMPLIANCE_MARGIN = 26
    ADDR_CCW_COMPLIANCE_MARGIN = 27
    ADDR_CW_COMPLIANCE_SLOPE = 28
    ADDR_CCW_COMPLIANCE_SLOPE = 29
    ADDR_GOAL_POSITION = 30
    ADDR_MOVING_SPEED = 32
    ADDR_PRESENT_POSITION = 36
    ADDR_PUNCH = 48
    
    # Torque constants
    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    
    # Robot DH parameters (in mm)
    L1 = 50.0   # base offset (d1)
    L2 = 93.0   # link 2 length (a2)
    L3 = 93.0   # link 3 length (a3)
    L4 = 50.0   # link 4 length (a4)
    
    def __init__(self, 
                 port_name: str = '/dev/ttyACM0',
                 baudrate: int = 1000000,
                 motor_ids: List[int] = [1, 2, 3, 4],
                 protocol_version: float = 1.0):
        """
        Initialize the robot controller.
        
        Args:
            port_name: Serial port name
            baudrate: Communication baudrate
            motor_ids: List of motor IDs to control
            protocol_version: Dynamixel protocol version
        """
        self.port_name = port_name
        self.baudrate = baudrate
        self.motor_ids = motor_ids
        self.protocol_version = protocol_version
        
        # Initialize handlers
        self.port_handler = dxl.PortHandler(self.port_name)
        self.packet_handler = dxl.PacketHandler(self.protocol_version)
        
        self.is_initialized = False
        self.is_connected = False

        self.joint_signs = [-1.0, 1.0, 1.0, 1.0]
        self.joint_offsets = [0.0, np.pi/2, 0.0, 0.0]
        
    def connect(self) -> bool:
        """
        Open the serial port and set baudrate.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if self.is_connected:
            print("Already connected!")
            return True
            
        # Open port
        if not self.port_handler.openPort():
            print(f"Failed to open port {self.port_name}")
            return False
        print(f"Successfully opened port {self.port_name}")
        
        # Set baudrate
        if not self.port_handler.setBaudRate(self.baudrate):
            print(f"Failed to set baudrate to {self.baudrate}")
            return False
        print(f"Successfully set baudrate to {self.baudrate}")
        
        self.is_connected = True
        return True
    
    def disconnect(self):
        """Close the serial port connection."""
        if self.is_connected:
            self.port_handler.closePort()
            self.is_connected = False
            print("Port closed successfully")
    
    def initialize(self, 
                   compliance_margin: int = 0,
                   compliance_slope: int = 32,
                   moving_speed: int = 50) -> bool:
        """
        Initialize all motors with torque enabled and compliance settings.
        
        Args:
            compliance_margin: Compliance margin value (0-255)
            compliance_slope: Compliance slope value (1-254)
            moving_speed: Moving speed value (0-1023)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not self.is_connected:
            print("Not connected! Call connect() first.")
            return False
        
        print(f"Initializing {len(self.motor_ids)} motors...")
        
        for motor_id in self.motor_ids:
            # Enable torque
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, 
                self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
            
            # Set compliance margins
            self.packet_handler.write2ByteTxRx(
                self.port_handler, motor_id,
                self.ADDR_CW_COMPLIANCE_MARGIN, compliance_margin
            )
            self.packet_handler.write2ByteTxRx(
                self.port_handler, motor_id,
                self.ADDR_CCW_COMPLIANCE_MARGIN, compliance_margin
            )
            
            # Set compliance slopes
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id,
                self.ADDR_CW_COMPLIANCE_SLOPE, compliance_slope
            )
            self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id,
                self.ADDR_CCW_COMPLIANCE_SLOPE, compliance_slope
            )
            
            # Set moving speed
            self.packet_handler.write2ByteTxRx(
                self.port_handler, motor_id,
                self.ADDR_MOVING_SPEED, moving_speed
            )
            
            print(f"  Motor {motor_id} initialized")
        
        self.is_initialized = True
        print("All motors initialized successfully!")
        return True
    
    def set_joint_positions(self, angles: List[float], wait: bool = False, timeout: float = 5.0) -> bool:
        """
        Set target positions for all joints using angles in radians.
        
        Args:
            angles: List of target angles in radians
            wait: If True, wait until motors reach target positions
            timeout: Maximum time to wait (seconds)
            
        Returns:
            bool: True if positions set successfully, False otherwise
        """
        if not self.is_initialized:
            print("Robot not initialized! Call initialize() first.")
            return False
        
        if len(angles) != len(self.motor_ids):
            print(f"Error: Expected {len(self.motor_ids)} angles, got {len(angles)}")
            return False
        
        # Convert radians to motor positions
        positions = self.radians_to_positions(np.array(angles))
        
        # Send goal positions
        for i, motor_id in enumerate(self.motor_ids):
            self.packet_handler.write2ByteTxRx(
                self.port_handler, motor_id,
                self.ADDR_GOAL_POSITION, positions[i]
            )
        
        # Wait for motors to reach positions if requested
        if wait:
            start_time = time.time()
            threshold = 10  # Position tolerance (in motor units)
            
            while True:
                current_positions_raw = self.get_joint_positions_raw()
                if current_positions_raw is None:
                    return False
                
                # Check if all motors reached target
                all_reached = all(
                    abs(current_positions_raw[i] - positions[i]) <= threshold
                    for i in range(len(positions))
                )
                
                if all_reached:
                    return True
                
                # Check timeout
                if time.time() - start_time > timeout:
                    print("Timeout waiting for motors to reach positions")
                    return False
                
                time.sleep(0.05)  # Small delay before next check
        
        return True
    
    def set_joint_positions_raw(self, positions: List[int], wait: bool = False, timeout: float = 5.0) -> bool:
        """
        Set target positions for all joints using raw motor positions.
        
        Args:
            positions: List of target positions (0-1023 for Protocol 1.0)
            wait: If True, wait until motors reach target positions
            timeout: Maximum time to wait (seconds)
            
        Returns:
            bool: True if positions set successfully, False otherwise
        """
        if not self.is_initialized:
            print("Robot not initialized! Call initialize() first.")
            return False
        
        if len(positions) != len(self.motor_ids):
            print(f"Error: Expected {len(self.motor_ids)} positions, got {len(positions)}")
            return False
        
        # Validate positions (0-1023 for Protocol 1.0)
        for i, pos in enumerate(positions):
            if pos < 0 or pos > 1023:
                print(f"Error: Position {pos} for motor {self.motor_ids[i]} out of range (0-1023)")
                return False
        
        # Send goal positions
        for i, motor_id in enumerate(self.motor_ids):
            self.packet_handler.write2ByteTxRx(
                self.port_handler, motor_id,
                self.ADDR_GOAL_POSITION, positions[i]
            )
        
        # Wait for motors to reach positions if requested
        if wait:
            start_time = time.time()
            threshold = 10  # Position tolerance
            
            while True:
                current_positions_raw = self.get_joint_positions_raw()
                if current_positions_raw is None:
                    return False
                
                # Check if all motors reached target
                all_reached = all(
                    abs(current_positions_raw[i] - positions[i]) <= threshold
                    for i in range(len(positions))
                )
                
                if all_reached:
                    return True
                
                # Check timeout
                if time.time() - start_time > timeout:
                    print("Timeout waiting for motors to reach positions")
                    return False
                
                time.sleep(0.05)  # Small delay before next check
        
        return True
    
    def get_joint_positions(self) -> Optional[np.ndarray]:
        """
        Read current positions of all joints in radians.
        
        Returns:
            np.ndarray: Current joint angles in radians, or None if read failed
        """
        if not self.is_connected:
            print("Not connected!")
            return None
        
        positions = []
        for motor_id in self.motor_ids:
            pos, comm_result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, self.ADDR_PRESENT_POSITION
            )
            
            if comm_result != dxl.COMM_SUCCESS:
                print(f"Failed to read position from motor {motor_id}")
                return None
            
            positions.append(pos)
        
        # Convert to radians
        return self.positions_to_radians(positions)
    
    def get_joint_positions_raw(self) -> Optional[List[int]]:
        """
        Read current positions of all joints as raw motor positions.
        
        Returns:
            List[int]: Current motor positions (0-1023), or None if read failed
        """
        if not self.is_connected:
            print("Not connected!")
            return None
        
        positions = []
        for motor_id in self.motor_ids:
            pos, comm_result, error = self.packet_handler.read2ByteTxRx(
                self.port_handler, motor_id, self.ADDR_PRESENT_POSITION
            )
            
            if comm_result != dxl.COMM_SUCCESS:
                print(f"Failed to read position from motor {motor_id}")
                return None
            
            positions.append(pos)
        
        return positions
    
    def set_single_joint(self, motor_index: int, angle: float, wait: bool = False) -> bool:
        """
        Set position for a single joint using angle in radians.
        
        Args:
            motor_index: Index of motor in motor_ids list (0-based)
            angle: Target angle in radians
            wait: If True, wait until motor reaches target position
            
        Returns:
            bool: True if position set successfully, False otherwise
        """
        if motor_index < 0 or motor_index >= len(self.motor_ids):
            print(f"Error: Motor index {motor_index} out of range")
            return False
        
        current_angles = self.get_joint_positions()
        if current_angles is None:
            return False
        
        current_angles[motor_index] = angle
        return self.set_joint_positions(list(current_angles), wait=wait)
    
    def enable_torque(self, motor_id: Optional[int] = None):
        """
        Enable torque for specified motor or all motors.
        
        Args:
            motor_id: Specific motor ID, or None for all motors
        """
        motors = [motor_id] if motor_id is not None else self.motor_ids
        
        for mid in motors:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, mid,
                self.ADDR_TORQUE_ENABLE, self.TORQUE_ENABLE
            )
        print(f"Torque enabled for motor(s): {motors}")
    
    def disable_torque(self, motor_id: Optional[int] = None):
        """
        Disable torque for specified motor or all motors.
        
        Args:
            motor_id: Specific motor ID, or None for all motors
        """
        motors = [motor_id] if motor_id is not None else self.motor_ids
        
        for mid in motors:
            self.packet_handler.write1ByteTxRx(
                self.port_handler, mid,
                self.ADDR_TORQUE_ENABLE, self.TORQUE_DISABLE
            )
        print(f"Torque disabled for motor(s): {motors}")
    
    def set_speed(self, speed: int, motor_id: Optional[int] = None):
        """
        Set moving speed for specified motor or all motors.
        
        Args:
            speed: Speed value (0-1023)
            motor_id: Specific motor ID, or None for all motors
        """
        motors = [motor_id] if motor_id is not None else self.motor_ids
        
        for mid in motors:
            self.packet_handler.write2ByteTxRx(
                self.port_handler, mid,
                self.ADDR_MOVING_SPEED, speed
            )
        print(f"Speed set to {speed} for motor(s): {motors}")
    
    # def inverse_kinematics(self, x: float, y: float, z: float, 
    #                       x4z_desired: float = 0.0, 
    #                       tol: float = 1e-9) -> List[np.ndarray]:
    #     """
    #     Compute inverse kinematics for the 4-DOF robot arm.
        
    #     This solves for joint angles (q1, q2, q3, q4) that place the end-effector
    #     at position (x, y, z) with the desired z-component of the x4 axis orientation.
        
    #     Args:
    #         x: Desired X position (mm)
    #         y: Desired Y position (mm)
    #         z: Desired Z position (mm)
    #         x4z_desired: Desired z-component of x4 axis (sin(q2+q3+q4)), range [-1, 1]
    #         tol: Tolerance for numerical computations
            
    #     Returns:
    #         List of possible joint configurations as numpy arrays [q1, q2, q3, q4] in radians.
    #         Returns empty list if position is unreachable.
    #     """
    #     # 1) Base joint from xy plane
    #     q1 = math.atan2(y, x)
    #     rho = math.hypot(x, y)  # horizontal distance
        
    #     # 2) Orientation scalar: (x4^0)_z = sin(S) where S = q2+q3+q4
    #     x4z_desired = float(np.clip(x4z_desired, -1.0, 1.0))
    #     S0 = math.asin(x4z_desired)
    #     # Two possible totals for S = q2+q3+q4
    #     S_candidates = (S0, math.pi - S0)
        
    #     solutions = []
        
    #     for S in S_candidates:
    #         # 3) Subtract the last link (L4) that is rotated by S
    #         rho_p = rho - self.L4 * math.cos(S)
    #         z_p = (z - self.L1) - self.L4 * math.sin(S)
            
    #         # 4) Planar 2R IK for links L2, L3
    #         num = rho_p**2 + z_p**2 - self.L2**2 - self.L3**2
    #         den = 2.0 * self.L2 * self.L3
            
    #         if abs(den) < tol:
    #             continue  # degenerate case
            
    #         c3 = num / den
            
    #         # Check reachability
    #         if c3 < -1.0 - 1e-6 or c3 > 1.0 + 1e-6:
    #             continue  # unreachable with this S
            
    #         c3 = max(min(c3, 1.0), -1.0)  # clamp to valid range
    #         s3_abs = math.sqrt(max(0.0, 1.0 - c3**2))
            
    #         # Two elbow configurations (up and down)
    #         for s3 in (s3_abs, -s3_abs):
    #             q3 = math.atan2(s3, c3)
                
    #             # Shoulder joint (standard 2R formula)
    #             q2 = math.atan2(z_p, rho_p) - math.atan2(
    #                 self.L3 * math.sin(q3),
    #                 self.L2 + self.L3 * math.cos(q3)
    #             )
                
    #             # 5) Recover q4 from S = q2 + q3 + q4
    #             q4 = S - (q2 + q3)
    #             # Joint limit for Dynamixel AX-12A (~150 degrees)
    #             LIMIT = 2.6
    #             if abs(q2) > LIMIT or abs(q3) > LIMIT or abs(q4) > LIMIT:
    #                 print("out of limits")
    #                 continue
    #             solutions.append(np.array([q1, q2, q3, q4], dtype=float))
    #     print(f"IK found {len(solutions)} solution(s) for position ({x}, {y}, {z})")
    #     print(f"Solutions (radians): {[np.round(sol, 4) for sol in solutions]}")
    #     return solutions
    
    def move_to_position(self, x: float, y: float, z: float,
                        x4z_desired: float = 0.0,
                        solution_index: int = 0,
                        wait: bool = True,
                        timeout: float = 5.0) -> bool:
        """
        Move robot end-effector to a target Cartesian position using IK.
        
        Args:
            x: Target X position (mm)
            y: Target Y position (mm)
            z: Target Z position (mm)
            x4z_desired: Desired z-component of x4 axis orientation (default: 0.0)
            solution_index: Which IK solution to use (0-3, if multiple exist)
            wait: Wait until motors reach target positions
            timeout: Maximum time to wait (seconds)
            
        Returns:
            bool: True if successful, False if position unreachable or error occurred
        """
        # Compute IK solutions
        solutions = self.inverse_kinematics(x, y, z, x4z_desired)
        
        # Ensure at least one valid (finite) solution exists
        if not any(np.isfinite(sol).all() for sol in solutions):
            print(f"Error: Position ({x}, {y}, {z}) is unreachable!")
            return False
        
        if solution_index >= len(solutions):
            print(f"Warning: Solution index {solution_index} not available. Using solution 0.")
            solution_index = 0
            print(len(solutions))
        
        # Get the desired joint angles in radians
        q_radians = solutions[solution_index]
        
        print(f"Moving to position ({x:.1f}, {y:.1f}, {z:.1f}) mm")
        print(f"Using IK solution {solution_index}: q = {np.round(np.degrees(q_radians), 2)} degrees")
        
        # Send positions to motors (now accepts radians directly)
        return self.set_joint_positions(list(q_radians), wait=wait, timeout=timeout)
    
    def radians_to_positions(self, q_radians: np.ndarray) -> List[int]:
        """Convert Math Angles (Radians) -> Servo Positions (0-1023)"""
        total_range_rads = math.radians(300.0)
        conversion_factor = 1023.0 / total_range_rads
        
        positions = []
        for i, angle in enumerate(q_radians):
            # BRIDGE: Math -> Physical
            # 1. Subtract Offset (Align Zero)
            # 2. Apply Sign (Align Direction)
            corrected_angle = (angle - self.joint_offsets[i]) * self.joint_signs[i]
            
            # 3. Convert to Ticks (Center 512)
            pos = int(round(512 + corrected_angle * conversion_factor))
            pos = max(0, min(1023, pos))
            positions.append(pos)
        return positions
    def inverse_kinematics(self, x: float, y: float, z: float, 
                          x4z_desired: float = 0.0, 
                          tol: float = 1e-9) -> List[np.ndarray]:
        """
        Compute inverse kinematics for the 4-DOF robot arm.
        Returns solutions for both Front and Back reach configurations.
        """
        solutions = []
        
        # 1) Calculate base parameters
        # Standard "Front" solution
        q1_front = math.atan2(y, x)
        rho_front = math.hypot(x, y)
        
        # "Back" solution (base rotated 180, arm reaches backwards)
        q1_back = q1_front + math.pi
        # Normalize angle to [-pi, pi]
        q1_back = (q1_back + math.pi) % (2 * math.pi) - math.pi
        rho_back = -rho_front
        
        # Define the sets of base configurations to test
        # format: (q1, horizontal_distance_to_target)
        base_configs = [(q1_front, rho_front), (q1_back, rho_back)]

        # 2) Orientation scalar
        x4z_desired = float(np.clip(x4z_desired, -1.0, 1.0))
        S0 = math.asin(x4z_desired)
        S_candidates = (S0, math.pi - S0)
        
        for q1, rho in base_configs:
            for S in S_candidates:
                # 3) Subtract the last link (L4)
                # Note: rho can be negative here for "back" solutions
                rho_p = rho - self.L4 * math.cos(S)
                z_p = (z - self.L1) - self.L4 * math.sin(S)
                
                # 4) Planar 2R IK for links L2, L3
                num = rho_p**2 + z_p**2 - self.L2**2 - self.L3**2
                den = 2.0 * self.L2 * self.L3
                
                if abs(den) < tol:
                    continue
                
                c3 = num / den
                
                if c3 < -1.0 - 1e-6 or c3 > 1.0 + 1e-6:
                    continue
                
                c3 = max(min(c3, 1.0), -1.0)
                s3_abs = math.sqrt(max(0.0, 1.0 - c3**2))
                
                # Two elbow configurations (up and down)
                for s3 in (s3_abs, -s3_abs):
                    q3 = math.atan2(s3, c3)
                    
                    # Shoulder joint
                    q2 = math.atan2(z_p, rho_p) - math.atan2(
                        self.L3 * math.sin(q3),
                        self.L2 + self.L3 * math.cos(q3)
                    )
                    
                    # 5) Recover q4
                    q4 = S - (q2 + q3)
                    
                    # Check Limits (approx +/- 150 degrees)
                    LIMIT = 2.6
                    if (abs(q2) > LIMIT or abs(q3) > LIMIT or abs(q4) > LIMIT):
                        continue
                        
                    solutions.append(np.array([q1, q2, q3, q4], dtype=float))

        # Optional: Sort solutions by "closeness" to current position to prevent jumps
        # You would need to pass current_joints to this function to enable this
        
        print(f"IK found {len(solutions)} solution(s) for position ({x}, {y}, {z})")
        return solutions

    def positions_to_radians(self, positions: List[int]) -> np.ndarray:
        """Convert Servo Positions (0-1023) -> Math Angles (Radians)"""
        total_range_rads = math.radians(300.0)
        conversion_factor = total_range_rads / 1023.0
        
        q_list = []
        for i, pos in enumerate(positions):
            # 1. Convert Ticks -> Raw Physical Radians
            raw_rad = (pos - 512) * conversion_factor
            
            # BRIDGE: Physical -> Math
            # 2. Apply Sign
            # 3. Add Offset
            math_rad = (raw_rad * self.joint_signs[i]) + self.joint_offsets[i]
            q_list.append(math_rad)
            
        return np.array(q_list)
    
    def forward_kinematics(self, q_radians: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics to get end-effector position and orientation.
        
        Args:
            q_radians: Joint angles in radians. If None, reads current positions.
            
        Returns:
            Tuple of (position, rotation_matrix):
                - position: [x, y, z] in mm
                - rotation_matrix: 3x3 rotation matrix of end-effector frame
        """
        if q_radians is None:
            q_radians = self.get_joint_positions()
            if q_radians is None:
                return None, None
        
        q1, q2, q3, q4 = q_radians
        
        # Compute transformation matrices using DH parameters
        # A1 = DH(q1, L1, 0, pi/2)
        c1, s1 = math.cos(q1), math.sin(q1)
        A1 = np.array([
            [c1, 0, s1, 0],
            [s1, 0, -c1, 0],
            [0, 1, 0, self.L1],
            [0, 0, 0, 1]
        ])
        
        # A2 = DH(q2, 0, L2, 0)
        c2, s2 = math.cos(q2), math.sin(q2)
        A2 = np.array([
            [c2, -s2, 0, self.L2*c2],
            [s2, c2, 0, self.L2*s2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # A3 = DH(q3, 0, L3, 0)
        c3, s3 = math.cos(q3), math.sin(q3)
        A3 = np.array([
            [c3, -s3, 0, self.L3*c3],
            [s3, c3, 0, self.L3*s3],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # A4 = DH(q4, 0, L4, 0)
        c4, s4 = math.cos(q4), math.sin(q4)
        A4 = np.array([
            [c4, -s4, 0, self.L4*c4],
            [s4, c4, 0, self.L4*s4],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Compute T0_4 = A1 @ A2 @ A3 @ A4
        T0_4 = A1 @ A2 @ A3 @ A4
        
        position = T0_4[:3, 3]
        rotation = T0_4[:3, :3]
        
        return position, rotation
    
    def get_camera_position(self, q_radians: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute forward kinematics to get camera position and orientation.
        
        Extends the stylus FK by applying the stylus-to-camera transformation.
        The transformation from stylus to camera is:
        T_stylus_camera = [[1, 0, 0, -15], [0, 1, 0, 45], [0, 0, 1, 0], [0, 0, 0, 1]]
        
        Args:
            q_radians: Joint angles in radians. If None, reads current positions.
            
        Returns:
            Tuple of (position, rotation_matrix):
                - position: [x, y, z] in mm (camera position in base frame)
                - rotation_matrix: 3x3 rotation matrix of camera frame
        """
        if q_radians is None:
            q_radians = self.get_joint_positions()
            if q_radians is None:
                return None, None
        
        q1, q2, q3, q4 = q_radians
        
        # Compute transformation matrices using DH parameters (base to stylus)
        # A1 = DH(q1, L1, 0, pi/2)
        c1, s1 = math.cos(q1), math.sin(q1)
        A1 = np.array([
            [c1, 0, s1, 0],
            [s1, 0, -c1, 0],
            [0, 1, 0, self.L1],
            [0, 0, 0, 1]
        ])
        
        # A2 = DH(q2, 0, L2, 0)
        c2, s2 = math.cos(q2), math.sin(q2)
        A2 = np.array([
            [c2, -s2, 0, self.L2*c2],
            [s2, c2, 0, self.L2*s2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # A3 = DH(q3, 0, L3, 0)
        c3, s3 = math.cos(q3), math.sin(q3)
        A3 = np.array([
            [c3, -s3, 0, self.L3*c3],
            [s3, c3, 0, self.L3*s3],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # A4 = DH(q4, 0, L4, 0)
        c4, s4 = math.cos(q4), math.sin(q4)
        A4 = np.array([
            [c4, -s4, 0, self.L4*c4],
            [s4, c4, 0, self.L4*s4],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Compute T_base_stylus = A1 @ A2 @ A3 @ A4
        T_base_stylus = A1 @ A2 @ A3 @ A4
        
        # Transformation from stylus to camera (A5)
        T_stylus_camera = np.array([
            [1, 0, 0, -15],
            [0, 1, 0, 45],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Complete transformation from base to camera
        T_base_camera = T_base_stylus @ T_stylus_camera
        
        position = T_base_camera[:3, 3]
        rotation = T_base_camera[:3, :3]
        
        return position, rotation
    
    def camera_to_base_frame(self, camera_coords: np.ndarray, q_radians: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform coordinates from camera frame to robot base frame.
        
        Uses the complete kinematic chain: Base -> Stylus -> Camera
        The transformation from stylus to camera is:
        T_stylus_camera = [[1, 0, 0, -15], [0, 1, 0, 45], [0, 0, 1, 0], [0, 0, 0, 1]]
        
        Args:
            camera_coords: 3D coordinates in camera frame [x, y, z] in mm
            q_radians: Joint angles in radians. If None, reads current positions.
            
        Returns:
            np.ndarray: 3D coordinates in robot base frame [x, y, z] in mm
        """
        if q_radians is None:
            q_radians = self.get_joint_positions()
            if q_radians is None:
                return None
        
        q1, q2, q3, q4 = q_radians
        
        # Compute transformation matrices using DH parameters (base to stylus)
        # A1 = DH(q1, L1, 0, pi/2)
        c1, s1 = math.cos(q1), math.sin(q1)
        A1 = np.array([
            [c1, 0, s1, 0],
            [s1, 0, -c1, 0],
            [0, 1, 0, self.L1],
            [0, 0, 0, 1]
        ])
        
        # A2 = DH(q2, 0, L2, 0)
        c2, s2 = math.cos(q2), math.sin(q2)
        A2 = np.array([
            [c2, -s2, 0, self.L2*c2],
            [s2, c2, 0, self.L2*s2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # A3 = DH(q3, 0, L3, 0)
        c3, s3 = math.cos(q3), math.sin(q3)
        A3 = np.array([
            [c3, -s3, 0, self.L3*c3],
            [s3, c3, 0, self.L3*s3],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # A4 = DH(q4, 0, L4, 0)
        c4, s4 = math.cos(q4), math.sin(q4)
        A4 = np.array([
            [c4, -s4, 0, self.L4*c4],
            [s4, c4, 0, self.L4*s4],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Compute T_base_stylus = A1 @ A2 @ A3 @ A4
        T_base_stylus = A1 @ A2 @ A3 @ A4
        
        # Transformation from stylus to camera (A5)
        T_stylus_camera = np.array([
            [1, 0, 0, -15],
            [0, 1, 0, 45],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=float)
        
        # Complete transformation from base to camera
        T_base_camera = T_base_stylus @ T_stylus_camera
        
        # Convert camera coordinates to homogeneous coordinates
        camera_coords_homogeneous = np.append(camera_coords, 1.0)
        
        # Transform to base frame
        # base_coords_homogeneous = T_base_camera @ camera_coords_homogeneous
        base_coords_homogeneous = T_base_stylus @ camera_coords_homogeneous
        
        # Return 3D coordinates (drop homogeneous coordinate)
        return base_coords_homogeneous[:3]
    
    def print_status(self):
        """Print current robot status."""
        print("\n" + "="*50)
        print("Robot Controller Status")
        print("="*50)
        print(f"Port: {self.port_name}")
        print(f"Baudrate: {self.baudrate}")
        print(f"Motor IDs: {self.motor_ids}")
        print(f"Protocol: {self.protocol_version}")
        print(f"Connected: {self.is_connected}")
        print(f"Initialized: {self.is_initialized}")
        
        if self.is_connected:
            angles = self.get_joint_positions()
            if angles is not None:
                print("\nCurrent Joint Angles:")
                for i, (motor_id, angle) in enumerate(zip(self.motor_ids, angles)):
                    print(f"  Joint {i+1} (Motor {motor_id}): {angle:.4f} rad ({np.degrees(angle):.2f}°)")
                
                # Calculate x4z component
                q1, q2, q3, q4 = angles
                x4z = math.sin(q2 + q3 + q4)
                print(f"\nx4z component: {x4z:.4f}")
                
                # Show Cartesian position
                pos_xyz, rot = self.forward_kinematics()
                if pos_xyz is not None:
                    print(f"\nEnd-Effector Position: [{pos_xyz[0]:.2f}, {pos_xyz[1]:.2f}, {pos_xyz[2]:.2f}] mm")
        print("="*50 + "\n")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup."""
        if self.is_initialized:
            self.disable_torque()
        self.disconnect()


# Example usage
if __name__ == "__main__":
    # Create robot controller
    robot = RobotController(
        port_name='/dev/ttyACM0',
        baudrate=1000000,
        motor_ids=[1, 2, 3, 4]
    )
    
    try:
        # Connect and initialize
        robot.connect()
        robot.initialize(compliance_margin=0, compliance_slope=32, moving_speed=100)
        
        # Print status
        robot.print_status()
        
        # Example 1: Move using joint positions (radians)
        print("\n=== Example 1: Joint Space Control ===")
        print("Moving to home position...")
        home_angles = [0, np.pi/2, 0, 0]  # All joints at 0 radians (center)
        # robot.set_joint_positions(home_angles, wait=True)
        # time.sleep(1)
        # robot.print_status()



        # Example 2: Move using Cartesian coordinates (IK)
        print("\n=== Example 2: Cartesian Space Control (IK) ===")
        
        # # Move to a reachable position
        print("\nMoving to position (150, 0, 120) mm...")
        x = 0.53
        y = 73.46
        z = 40.43

        # if robot.move_to_position(x, y, z, x4z_desired=-1, wait=False, solution_index=0):
        #     time.sleep(5)
        #     robot.print_status()
        # if robot.move_to_position(x+30, y, z, x4z_desired=-1, wait=False, solution_index=0):
        #     time.sleep(5)
        #     robot.print_status()
        # if robot.move_to_position(x+30, y+30, z, x4z_desired=-1, wait=False, solution_index=0):
        #     time.sleep(5)
        #     robot.print_status()
        # if robot.move_to_position(x, y, z, x4z_desired=-1, wait=False, solution_index=1):
        #     time.sleep(5)
        #     robot.print_status()
        
        # if robot.move_to_position(x, y, z, x4z_desired=1, wait=False):
        #     time.sleep(5)
        #     robot.print_status()


        # if robot.move_to_position(158.79, -1.63, 166.90, x4z_desired=0, wait=True):
        #         time.sleep(1)
        #         robot.print_status()
        #         exit()
        # else:
        #     print("Failed to move to initial position.")
        #     robot.print_status()
        #     exit()

        for i in range(45):
            if robot.move_to_position(x+i, y, z, x4z_desired=-1, wait=False):
                time.sleep(0.1)
                robot.print_status()


        
        
        # Move through a circular path
        # print("\n=== Example 3: Circular Path ===")
        # R = 32  # radius in mm
        # pc = np.array([150, 0, 120])  # center point
        
        # for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
        #     x = pc[0]
        #     y = pc[1] + R * np.cos(angle)
        #     z = pc[2] + R * np.sin(angle)
            
        #     print(f"\nMoving to ({x:.1f}, {y:.1f}, {z:.1f}) mm...")
        #     if robot.move_to_position(x, y, z, x4z_desired=0.0, solution_index=0, wait=True):
        #         time.sleep(0.5)
        
        print("\n=== Returning to home ===")
        # robot.set_joint_positions([0.0, 0.0, np.pi/2, 0.0], wait=True)
        
        print("\nSequence complete!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        robot.disable_torque()
        robot.disconnect()

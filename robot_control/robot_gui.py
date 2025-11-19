#!/usr/bin/env python3
"""
Robot Controller GUI
A PyQt5-based graphical interface for controlling the robot arm.
"""

import sys
import os

# Fix Qt plugin conflict between OpenCV and PyQt5
# Must be set before importing any Qt or cv2 modules
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
if "cv2" in sys.modules:
    del sys.modules["cv2"]

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QDoubleSpinBox,
    QSpinBox, QTabWidget, QGridLayout, QMessageBox, QSlider, QFileDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QImage, QPixmap
from robot_controller import RobotController
import math
# Import cv2 with headless backend to avoid Qt conflicts
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
from datetime import datetime
from object_detection import ObjectDetector


class RobotGUI(QMainWindow):
    """Main GUI window for robot control."""
    
    def __init__(self):
        super().__init__()
        self.robot = None
        self.realtime_update_enabled = False
        self.camera = None
        self.camera_active = False
        self.detector = ObjectDetector(calibration_file='./photos/camera_calibration_data.npz')
        self.detection_enabled = False
        self.current_detections = []
        # Camera to stylus transformation (inverse of stylus to camera)
        # Original: T_stylus_camera = [[1,0,0,-15], [0,1,0,45], [0,0,1,0], [0,0,0,1]]
        # Inverted: T_camera_stylus
        self.camera_to_stylus_offset = np.array([15.0, -45.0, 0.0])  # mm
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Robot Controller GUI")
        self.setGeometry(100, 100, 900, 700)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Connection section
        connection_group = self.create_connection_group()
        main_layout.addWidget(connection_group)
        
        # Create tab widget for different control modes
        tab_widget = QTabWidget()
        
        # Joint Control Tab
        joint_tab = self.create_joint_control_tab()
        tab_widget.addTab(joint_tab, "Joint Control")
        
        # Cartesian Control Tab
        cartesian_tab = self.create_cartesian_control_tab()
        tab_widget.addTab(cartesian_tab, "Cartesian Control (IK)")
        
        # Status Tab
        status_tab = self.create_status_tab()
        tab_widget.addTab(status_tab, "Status")
        
        # Real-time Control Tab
        realtime_tab = self.create_realtime_control_tab()
        tab_widget.addTab(realtime_tab, "Real-time Control")
        
        # Camera Tab
        camera_tab = self.create_camera_tab()
        tab_widget.addTab(camera_tab, "Camera")
        
        main_layout.addWidget(tab_widget)
        
        # Log output
        log_group = QGroupBox("Log Output")
        log_layout = QVBoxLayout()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        log_layout.addWidget(self.log_output)
        log_group.setLayout(log_layout)
        main_layout.addWidget(log_group)
        
        self.log("GUI initialized. Connect to robot to begin.")
        
    def create_connection_group(self):
        """Create connection control group."""
        group = QGroupBox("Connection Settings")
        layout = QHBoxLayout()
        
        # Port name
        layout.addWidget(QLabel("Port:"))
        self.port_input = QLineEdit("/dev/ttyACM0")
        self.port_input.setMaximumWidth(150)
        layout.addWidget(self.port_input)
        
        # Baudrate
        layout.addWidget(QLabel("Baudrate:"))
        self.baudrate_input = QLineEdit("1000000")
        self.baudrate_input.setMaximumWidth(100)
        layout.addWidget(self.baudrate_input)
        
        # Motor IDs
        layout.addWidget(QLabel("Motor IDs:"))
        self.motor_ids_input = QLineEdit("1,2,3,4")
        self.motor_ids_input.setMaximumWidth(100)
        layout.addWidget(self.motor_ids_input)
        
        # Connect button
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_robot)
        layout.addWidget(self.connect_btn)
        
        # Disconnect button
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.disconnect_robot)
        self.disconnect_btn.setEnabled(False)
        layout.addWidget(self.disconnect_btn)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
        
    def create_joint_control_tab(self):
        """Create joint control tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Joint angle inputs
        joint_group = QGroupBox("Joint Angles (degrees)")
        joint_layout = QGridLayout()
        
        self.joint_spinboxes = []
        for i in range(4):
            label = QLabel(f"Joint {i+1}:")
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-180, 180)
            spinbox.setSingleStep(1.0)
            spinbox.setDecimals(2)
            spinbox.setValue(0.0)
            spinbox.setWrapping(False)
            spinbox.setKeyboardTracking(True)
            spinbox.setSuffix("°")
            self.joint_spinboxes.append(spinbox)
            
            joint_layout.addWidget(label, i, 0)
            joint_layout.addWidget(spinbox, i, 1)
            
            # Add radian display
            rad_label = QLabel("0.0000 rad")
            spinbox.valueChanged.connect(
                lambda val, lbl=rad_label: lbl.setText(f"{np.radians(val):.4f} rad")
            )
            joint_layout.addWidget(rad_label, i, 2)
        
        joint_group.setLayout(joint_layout)
        layout.addWidget(joint_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        move_joint_btn = QPushButton("Move to Joint Angles")
        move_joint_btn.clicked.connect(self.move_joints)
        btn_layout.addWidget(move_joint_btn)
        
        move_joint_wait_btn = QPushButton("Move and Wait")
        move_joint_wait_btn.clicked.connect(lambda: self.move_joints(wait=True))
        btn_layout.addWidget(move_joint_wait_btn)
        
        read_joint_btn = QPushButton("Read Current Angles")
        read_joint_btn.clicked.connect(self.read_joint_angles)
        btn_layout.addWidget(read_joint_btn)
        
        layout.addLayout(btn_layout)
        
        # Preset positions
        preset_group = QGroupBox("Preset Positions")
        preset_layout = QHBoxLayout()
        
        home_btn = QPushButton("Home (All Zero)")
        home_btn.clicked.connect(lambda: self.set_preset([0.0, 0.0, 0.0, 0.0]))
        preset_layout.addWidget(home_btn)
        
        preset_1_btn = QPushButton("Preset 1")
        preset_1_btn.clicked.connect(lambda: self.set_preset([-np.pi/2, np.pi/2, np.pi/2, 0.0]))
        preset_layout.addWidget(preset_1_btn)
        
        preset_2_btn = QPushButton("Preset 2")
        preset_2_btn.clicked.connect(lambda: self.set_preset([0.0, np.pi/4, np.pi/4, np.pi/4]))
        preset_layout.addWidget(preset_2_btn)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_cartesian_control_tab(self):
        """Create Cartesian control tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Position inputs
        position_group = QGroupBox("Target Position (mm)")
        position_layout = QGridLayout()
        
        # X, Y, Z inputs
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(-300, 300)
        self.x_spinbox.setSingleStep(1.0)
        self.x_spinbox.setDecimals(2)
        self.x_spinbox.setValue(100.0)
        
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(-300, 300)
        self.y_spinbox.setSingleStep(1.0)
        self.y_spinbox.setDecimals(2)
        self.y_spinbox.setValue(0.0)
        
        self.z_spinbox = QDoubleSpinBox()
        self.z_spinbox.setRange(0, 300)
        self.z_spinbox.setSingleStep(1.0)
        self.z_spinbox.setDecimals(2)
        self.z_spinbox.setValue(150.0)
        
        position_layout.addWidget(QLabel("X:"), 0, 0)
        position_layout.addWidget(self.x_spinbox, 0, 1)
        position_layout.addWidget(QLabel("Y:"), 1, 0)
        position_layout.addWidget(self.y_spinbox, 1, 1)
        position_layout.addWidget(QLabel("Z:"), 2, 0)
        position_layout.addWidget(self.z_spinbox, 2, 1)
        
        position_group.setLayout(position_layout)
        layout.addWidget(position_group)
        
        # Orientation input
        orientation_group = QGroupBox("Orientation")
        orientation_layout = QGridLayout()
        
        self.x4z_spinbox = QDoubleSpinBox()
        self.x4z_spinbox.setRange(-1.0, 1.0)
        self.x4z_spinbox.setSingleStep(0.1)
        self.x4z_spinbox.setDecimals(2)
        self.x4z_spinbox.setValue(0.0)
        
        orientation_layout.addWidget(QLabel("x4z component:"), 0, 0)
        orientation_layout.addWidget(self.x4z_spinbox, 0, 1)
        orientation_layout.addWidget(QLabel("(sin(q2+q3+q4))"), 0, 2)
        
        orientation_group.setLayout(orientation_layout)
        layout.addWidget(orientation_group)
        
        # Solution index
        solution_group = QGroupBox("IK Solution Selection")
        solution_layout = QHBoxLayout()
        
        solution_layout.addWidget(QLabel("Solution Index:"))
        self.solution_spinbox = QSpinBox()
        self.solution_spinbox.setRange(0, 3)
        self.solution_spinbox.setValue(0)
        solution_layout.addWidget(self.solution_spinbox)
        solution_layout.addStretch()
        
        solution_group.setLayout(solution_layout)
        layout.addWidget(solution_group)
        
        # Control buttons
        btn_layout = QHBoxLayout()
        
        move_ik_btn = QPushButton("Move to Position (IK)")
        move_ik_btn.clicked.connect(self.move_cartesian)
        btn_layout.addWidget(move_ik_btn)
        
        move_ik_wait_btn = QPushButton("Move and Wait")
        move_ik_wait_btn.clicked.connect(lambda: self.move_cartesian(wait=True))
        btn_layout.addWidget(move_ik_wait_btn)
        
        compute_ik_btn = QPushButton("Compute IK (Preview)")
        compute_ik_btn.clicked.connect(self.compute_ik_preview)
        btn_layout.addWidget(compute_ik_btn)
        
        layout.addLayout(btn_layout)
        
        # FK button to get current position
        fk_btn = QPushButton("Get Current End-Effector Position (FK)")
        fk_btn.clicked.connect(self.compute_forward_kinematics)
        layout.addWidget(fk_btn)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_status_tab(self):
        """Create status tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Status display
        status_group = QGroupBox("Robot Status")
        status_layout = QVBoxLayout()
        
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMinimumHeight(300)
        font = QFont("Courier New", 10)
        self.status_display.setFont(font)
        status_layout.addWidget(self.status_display)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Refresh button
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Status")
        refresh_btn.clicked.connect(self.update_status)
        btn_layout.addWidget(refresh_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        # Motor control
        motor_group = QGroupBox("Motor Control")
        motor_layout = QHBoxLayout()
        
        enable_btn = QPushButton("Enable Torque")
        enable_btn.clicked.connect(self.enable_torque)
        motor_layout.addWidget(enable_btn)
        
        disable_btn = QPushButton("Disable Torque")
        disable_btn.clicked.connect(self.disable_torque)
        motor_layout.addWidget(disable_btn)
        
        # Speed control
        motor_layout.addWidget(QLabel("Speed:"))
        self.speed_spinbox = QSpinBox()
        self.speed_spinbox.setRange(0, 1023)
        self.speed_spinbox.setValue(100)
        motor_layout.addWidget(self.speed_spinbox)
        
        set_speed_btn = QPushButton("Set Speed")
        set_speed_btn.clicked.connect(self.set_speed)
        motor_layout.addWidget(set_speed_btn)
        
        motor_group.setLayout(motor_layout)
        layout.addWidget(motor_group)
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def create_realtime_control_tab(self):
        """Create real-time control tab with sliders."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Enable/Disable real-time control
        control_layout = QHBoxLayout()
        self.realtime_enable_btn = QPushButton("Enable Real-time Control")
        self.realtime_enable_btn.clicked.connect(self.toggle_realtime_control)
        control_layout.addWidget(self.realtime_enable_btn)
        
        sync_btn = QPushButton("Sync Sliders to Current Position")
        sync_btn.clicked.connect(self.sync_sliders_to_robot)
        control_layout.addWidget(sync_btn)
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Joint angle sliders
        joint_slider_group = QGroupBox("Joint Angle Control (Real-time)")
        joint_slider_layout = QGridLayout()
        
        self.joint_sliders = []
        self.joint_slider_labels = []
        
        for i in range(4):
            label = QLabel(f"Joint {i+1}:")
            joint_slider_layout.addWidget(label, i, 0)
            
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-314, 314)  # -pi to +pi in hundredths of radians (*100)
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(157)  # pi/2 intervals
            slider.valueChanged.connect(lambda val, idx=i: self.on_joint_slider_changed(idx, val))
            self.joint_sliders.append(slider)
            joint_slider_layout.addWidget(slider, i, 1)
            
            value_label = QLabel("0.00 rad (0.00°)")
            self.joint_slider_labels.append(value_label)
            joint_slider_layout.addWidget(value_label, i, 2)
        
        joint_slider_group.setLayout(joint_slider_layout)
        layout.addWidget(joint_slider_group)
        
        # Cartesian position sliders
        cartesian_slider_group = QGroupBox("Cartesian Position Control (Real-time IK)")
        cartesian_slider_layout = QGridLayout()
        
        # X slider
        cartesian_slider_layout.addWidget(QLabel("X (mm):"), 0, 0)
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setRange(-300, 300)
        self.x_slider.setValue(100)
        self.x_slider.setTickPosition(QSlider.TicksBelow)
        self.x_slider.setTickInterval(50)
        self.x_slider.valueChanged.connect(lambda val: self.on_cartesian_slider_changed('x', val))
        cartesian_slider_layout.addWidget(self.x_slider, 0, 1)
        self.x_slider_label = QLabel("100 mm")
        cartesian_slider_layout.addWidget(self.x_slider_label, 0, 2)
        
        # Y slider
        cartesian_slider_layout.addWidget(QLabel("Y (mm):"), 1, 0)
        self.y_slider = QSlider(Qt.Horizontal)
        self.y_slider.setRange(-300, 300)
        self.y_slider.setValue(0)
        self.y_slider.setTickPosition(QSlider.TicksBelow)
        self.y_slider.setTickInterval(50)
        self.y_slider.valueChanged.connect(lambda val: self.on_cartesian_slider_changed('y', val))
        cartesian_slider_layout.addWidget(self.y_slider, 1, 1)
        self.y_slider_label = QLabel("0 mm")
        cartesian_slider_layout.addWidget(self.y_slider_label, 1, 2)
        
        # Z slider
        cartesian_slider_layout.addWidget(QLabel("Z (mm):"), 2, 0)
        self.z_slider = QSlider(Qt.Horizontal)
        self.z_slider.setRange(0, 300)
        self.z_slider.setValue(150)
        self.z_slider.setTickPosition(QSlider.TicksBelow)
        self.z_slider.setTickInterval(50)
        self.z_slider.valueChanged.connect(lambda val: self.on_cartesian_slider_changed('z', val))
        cartesian_slider_layout.addWidget(self.z_slider, 2, 1)
        self.z_slider_label = QLabel("150 mm")
        cartesian_slider_layout.addWidget(self.z_slider_label, 2, 2)
        
        # x4z orientation slider
        cartesian_slider_layout.addWidget(QLabel("x4z:"), 3, 0)
        self.x4z_slider = QSlider(Qt.Horizontal)
        self.x4z_slider.setRange(-100, 100)  # -1.0 to 1.0 in hundredths
        self.x4z_slider.setValue(0)
        self.x4z_slider.setTickPosition(QSlider.TicksBelow)
        self.x4z_slider.setTickInterval(25)
        self.x4z_slider.valueChanged.connect(lambda val: self.on_cartesian_slider_changed('x4z', val))
        cartesian_slider_layout.addWidget(self.x4z_slider, 3, 1)
        self.x4z_slider_label = QLabel("0.00")
        cartesian_slider_layout.addWidget(self.x4z_slider_label, 3, 2)
        
        # Solution index selector
        cartesian_slider_layout.addWidget(QLabel("Solution:"), 4, 0)
        self.realtime_solution_spinbox = QSpinBox()
        self.realtime_solution_spinbox.setRange(0, 7)
        self.realtime_solution_spinbox.setValue(0)
        cartesian_slider_layout.addWidget(self.realtime_solution_spinbox, 4, 1)
        
        cartesian_slider_group.setLayout(cartesian_slider_layout)
        layout.addWidget(cartesian_slider_group)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Control Mode:"))
        self.joint_mode_btn = QPushButton("Joint Mode")
        self.joint_mode_btn.setCheckable(True)
        self.joint_mode_btn.setChecked(True)
        self.joint_mode_btn.clicked.connect(lambda: self.set_realtime_mode('joint'))
        mode_layout.addWidget(self.joint_mode_btn)
        
        self.cartesian_mode_btn = QPushButton("Cartesian Mode")
        self.cartesian_mode_btn.setCheckable(True)
        self.cartesian_mode_btn.clicked.connect(lambda: self.set_realtime_mode('cartesian'))
        mode_layout.addWidget(self.cartesian_mode_btn)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)
        
        self.realtime_mode = 'joint'  # Default mode
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
    
    def create_camera_tab(self):
        """Create camera control tab."""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Camera control
        camera_control_group = QGroupBox("Camera Control")
        camera_control_layout = QHBoxLayout()
        
        # Camera device selector
        camera_control_layout.addWidget(QLabel("Camera Device:"))
        self.camera_device_spinbox = QSpinBox()
        self.camera_device_spinbox.setRange(0, 10)
        self.camera_device_spinbox.setValue(2)
        camera_control_layout.addWidget(self.camera_device_spinbox)
        
        detect_btn = QPushButton("Detect Cameras")
        detect_btn.clicked.connect(self.detect_cameras)
        camera_control_layout.addWidget(detect_btn)
        
        self.camera_open_btn = QPushButton("Open Camera")
        self.camera_open_btn.clicked.connect(self.open_camera)
        camera_control_layout.addWidget(self.camera_open_btn)
        
        self.camera_close_btn = QPushButton("Close Camera")
        self.camera_close_btn.clicked.connect(self.close_camera)
        self.camera_close_btn.setEnabled(False)
        camera_control_layout.addWidget(self.camera_close_btn)
        
        camera_control_layout.addStretch()
        camera_control_group.setLayout(camera_control_layout)
        layout.addWidget(camera_control_group)
        
        # Detection control
        detection_control_group = QGroupBox("Object Detection")
        detection_control_layout = QHBoxLayout()
        
        self.detection_toggle_btn = QPushButton("Enable Detection")
        self.detection_toggle_btn.clicked.connect(self.toggle_detection)
        self.detection_toggle_btn.setEnabled(False)
        detection_control_layout.addWidget(self.detection_toggle_btn)
        
        detection_control_layout.addWidget(QLabel("Camera height is automatically set from robot position"))
        
        detection_control_layout.addStretch()
        detection_control_group.setLayout(detection_control_layout)
        layout.addWidget(detection_control_group)
        
        # Detection list
        detection_list_group = QGroupBox("Detected Objects")
        detection_list_layout = QVBoxLayout()
        
        self.detection_list = QTextEdit()
        self.detection_list.setReadOnly(True)
        self.detection_list.setMaximumHeight(150)
        font = QFont("Courier New", 9)
        self.detection_list.setFont(font)
        self.detection_list.setText("No detections yet")
        detection_list_layout.addWidget(self.detection_list)
        
        # Move to detection controls
        move_detection_layout = QHBoxLayout()
        move_detection_layout.addWidget(QLabel("Move to Detection #:"))
        self.detection_index_spinbox = QSpinBox()
        self.detection_index_spinbox.setRange(1, 10)
        self.detection_index_spinbox.setValue(1)
        move_detection_layout.addWidget(self.detection_index_spinbox)
        
        move_to_detection_btn = QPushButton("Move Stylus to Detection")
        move_to_detection_btn.clicked.connect(self.move_to_detection)
        move_detection_layout.addWidget(move_to_detection_btn)
        
        move_detection_layout.addStretch()
        detection_list_layout.addLayout(move_detection_layout)
        
        detection_list_group.setLayout(detection_list_layout)
        layout.addWidget(detection_list_group)
        
        # Camera preview
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout()
        
        self.camera_preview_label = QLabel("Camera not active")
        self.camera_preview_label.setAlignment(Qt.AlignCenter)
        self.camera_preview_label.setMinimumSize(640, 480)
        self.camera_preview_label.setStyleSheet("border: 1px solid black; background-color: #2b2b2b;")
        preview_layout.addWidget(self.camera_preview_label)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Photo capture
        capture_group = QGroupBox("Photo Capture")
        capture_layout = QHBoxLayout()
        
        self.capture_btn = QPushButton("Take Photo")
        self.capture_btn.clicked.connect(self.take_photo)
        self.capture_btn.setEnabled(False)
        capture_layout.addWidget(self.capture_btn)
        
        self.save_location_label = QLabel("Save to: ./photos/")
        capture_layout.addWidget(self.save_location_label)
        
        change_location_btn = QPushButton("Change Location")
        change_location_btn.clicked.connect(self.change_save_location)
        capture_layout.addWidget(change_location_btn)
        
        capture_layout.addStretch()
        capture_group.setLayout(capture_layout)
        layout.addWidget(capture_group)
        
        # Set up timer for camera updates
        self.camera_timer = QTimer()
        self.camera_timer.timeout.connect(self.update_camera_frame)
        
        # Default save location
        self.save_location = "./photos"
        
        layout.addStretch()
        widget.setLayout(layout)
        return widget
        
    def toggle_realtime_control(self):
        """Enable or disable real-time control."""
        if not self.robot or not self.robot.is_initialized:
            self.log("Robot not connected or initialized!")
            QMessageBox.warning(self, "Not Connected", "Please connect and initialize the robot first.")
            return
        
        self.realtime_update_enabled = not self.realtime_update_enabled
        
        if self.realtime_update_enabled:
            self.realtime_enable_btn.setText("Disable Real-time Control")
            self.realtime_enable_btn.setStyleSheet("background-color: #ff6b6b;")
            self.log("Real-time control ENABLED - Sliders will move robot immediately!")
            self.sync_sliders_to_robot()
        else:
            self.realtime_enable_btn.setText("Enable Real-time Control")
            self.realtime_enable_btn.setStyleSheet("")
            self.log("Real-time control DISABLED")
    
    def set_realtime_mode(self, mode):
        """Set the real-time control mode (joint or cartesian)."""
        self.realtime_mode = mode
        if mode == 'joint':
            self.joint_mode_btn.setChecked(True)
            self.cartesian_mode_btn.setChecked(False)
            self.log("Switched to Joint Mode")
        else:
            self.joint_mode_btn.setChecked(False)
            self.cartesian_mode_btn.setChecked(True)
            self.log("Switched to Cartesian Mode")
            if self.realtime_update_enabled:
                self.sync_cartesian_sliders_to_robot()
    
    def sync_sliders_to_robot(self):
        """Sync all sliders to current robot position."""
        if not self.robot or not self.robot.is_connected:
            self.log("Robot not connected!")
            return
        
        try:
            # Sync joint sliders
            angles = self.robot.get_joint_positions()
            if angles is not None:
                for i, angle in enumerate(angles):
                    # Temporarily disable signals to avoid triggering movements
                    self.joint_sliders[i].blockSignals(True)
                    self.joint_sliders[i].setValue(int(angle * 100))
                    self.joint_slider_labels[i].setText(f"{angle:.2f} rad ({np.degrees(angle):.2f}°)")
                    self.joint_sliders[i].blockSignals(False)
                
                # Sync cartesian sliders
                self.sync_cartesian_sliders_to_robot()
                
                self.log("Sliders synchronized to robot position")
        except Exception as e:
            self.log(f"Error syncing sliders: {str(e)}")
    
    def sync_cartesian_sliders_to_robot(self):
        """Sync cartesian sliders to current end-effector position."""
        if not self.robot or not self.robot.is_connected:
            return
        
        try:
            pos, rot = self.robot.forward_kinematics()
            if pos is not None:
                self.x_slider.blockSignals(True)
                self.y_slider.blockSignals(True)
                self.z_slider.blockSignals(True)
                self.x4z_slider.blockSignals(True)
                
                self.x_slider.setValue(int(pos[0]))
                self.y_slider.setValue(int(pos[1]))
                self.z_slider.setValue(int(pos[2]))
                
                self.x_slider_label.setText(f"{int(pos[0])} mm")
                self.y_slider_label.setText(f"{int(pos[1])} mm")
                self.z_slider_label.setText(f"{int(pos[2])} mm")
                
                # Calculate x4z from current joint angles
                angles = self.robot.get_joint_positions()
                if angles is not None:
                    q1, q2, q3, q4 = angles
                    x4z = math.sin(q2 + q3 + q4)
                    self.x4z_slider.setValue(int(x4z * 100))
                    self.x4z_slider_label.setText(f"{x4z:.2f}")
                
                self.x_slider.blockSignals(False)
                self.y_slider.blockSignals(False)
                self.z_slider.blockSignals(False)
                self.x4z_slider.blockSignals(False)
        except Exception as e:
            self.log(f"Error syncing cartesian sliders: {str(e)}")
    
    def on_joint_slider_changed(self, joint_idx, value):
        """Handle joint slider changes."""
        angle_rad = value / 100.0
        self.joint_slider_labels[joint_idx].setText(f"{angle_rad:.2f} rad ({np.degrees(angle_rad):.2f}°)")
        
        if self.realtime_update_enabled and self.realtime_mode == 'joint':
            if not self.robot or not self.robot.is_initialized:
                return
            
            try:
                # Get all current slider values
                angles = [slider.value() / 100.0 for slider in self.joint_sliders]
                self.robot.set_joint_positions(angles, wait=False)
                
                # Update cartesian sliders to reflect new position
                QTimer.singleShot(100, self.sync_cartesian_sliders_to_robot)
            except Exception as e:
                self.log(f"Error in real-time joint control: {str(e)}")
    
    def on_cartesian_slider_changed(self, axis, value):
        """Handle cartesian slider changes."""
        if axis == 'x':
            self.x_slider_label.setText(f"{value} mm")
        elif axis == 'y':
            self.y_slider_label.setText(f"{value} mm")
        elif axis == 'z':
            self.z_slider_label.setText(f"{value} mm")
        elif axis == 'x4z':
            x4z_val = value / 100.0
            self.x4z_slider_label.setText(f"{x4z_val:.2f}")
        
        if self.realtime_update_enabled and self.realtime_mode == 'cartesian':
            if not self.robot or not self.robot.is_initialized:
                return
            
            try:
                x = self.x_slider.value()
                y = self.y_slider.value()
                z = self.z_slider.value()
                x4z = self.x4z_slider.value() / 100.0
                solution_idx = self.realtime_solution_spinbox.value()
                
                # Compute IK and move
                solutions = self.robot.inverse_kinematics(x, y, z, x4z_desired=x4z)
                if solutions and solution_idx < len(solutions):
                    q_radians = solutions[solution_idx]
                    self.robot.set_joint_positions(list(q_radians), wait=False)
                    
                    # Update joint sliders to reflect new angles
                    for i, angle in enumerate(q_radians):
                        self.joint_sliders[i].blockSignals(True)
                        self.joint_sliders[i].setValue(int(angle * 100))
                        self.joint_slider_labels[i].setText(f"{angle:.2f} rad ({np.degrees(angle):.2f}°)")
                        self.joint_sliders[i].blockSignals(False)
            except Exception as e:
                self.log(f"Error in real-time cartesian control: {str(e)}")
        
    def log(self, message):
        """Add message to log output."""
        self.log_output.append(message)
        
    def connect_robot(self):
        """Connect to the robot."""
        try:
            port = self.port_input.text()
            baudrate = int(self.baudrate_input.text())
            motor_ids = [int(x.strip()) for x in self.motor_ids_input.text().split(',')]
            
            self.log(f"Connecting to robot at {port}...")
            self.robot = RobotController(
                port_name=port,
                baudrate=baudrate,
                motor_ids=motor_ids,
                protocol_version=1.0
            )
            
            if self.robot.connect():
                self.log("Connected successfully!")
                if self.robot.initialize(compliance_margin=0, compliance_slope=32, moving_speed=50):
                    self.log("Robot initialized successfully!")
                    self.connect_btn.setEnabled(False)
                    self.disconnect_btn.setEnabled(True)
                    self.update_status()
                else:
                    self.log("Failed to initialize robot.")
            else:
                self.log("Failed to connect to robot.")
                self.robot = None
                
        except Exception as e:
            self.log(f"Error connecting: {str(e)}")
            QMessageBox.critical(self, "Connection Error", str(e))
            
    def disconnect_robot(self):
        """Disconnect from the robot."""
        if self.robot:
            self.log("Disconnecting from robot...")
            self.robot.disable_torque()
            self.robot.disconnect()
            self.robot = None
            self.connect_btn.setEnabled(True)
            self.disconnect_btn.setEnabled(False)
            self.log("Disconnected.")
            
    def move_joints(self, wait=False):
        """Move robot to specified joint angles."""
        if not self.robot or not self.robot.is_initialized:
            self.log("Robot not connected or initialized!")
            return
            
        try:
            angles_deg = [spinbox.value() for spinbox in self.joint_spinboxes]
            angles_rad = [np.radians(deg) for deg in angles_deg]
            self.log(f"Moving to joint angles: {[f'{a:.2f}°' for a in angles_deg]}")
            
            if self.robot.set_joint_positions(angles_rad, wait=wait):
                self.log("Joint movement command sent successfully.")
            else:
                self.log("Failed to send joint movement command.")
                
        except Exception as e:
            self.log(f"Error moving joints: {str(e)}")
            QMessageBox.critical(self, "Movement Error", str(e))
            
    def read_joint_angles(self):
        """Read current joint angles from robot."""
        if not self.robot or not self.robot.is_connected:
            self.log("Robot not connected!")
            return
            
        try:
            angles_rad = self.robot.get_joint_positions()
            if angles_rad is not None:
                angles_deg = [np.degrees(a) for a in angles_rad]
                for i, (spinbox, angle_deg) in enumerate(zip(self.joint_spinboxes, angles_deg)):
                    spinbox.setValue(angle_deg)
                self.log(f"Read joint angles: {[f'{a:.2f}°' for a in angles_deg]}")
            else:
                self.log("Failed to read joint angles.")
                
        except Exception as e:
            self.log(f"Error reading joint angles: {str(e)}")
            
    def set_preset(self, angles_rad):
        """Set spinboxes to preset angles (input in radians, converted to degrees for display)."""
        angles_deg = [np.degrees(a) for a in angles_rad]
        for spinbox, angle_deg in zip(self.joint_spinboxes, angles_deg):
            spinbox.setValue(angle_deg)
        self.log(f"Set preset angles: {[f'{a:.2f}°' for a in angles_deg]}")
        
    def move_cartesian(self, wait=False):
        """Move robot to specified Cartesian position using IK."""
        if not self.robot or not self.robot.is_initialized:
            self.log("Robot not connected or initialized!")
            return
            
        try:
            x = self.x_spinbox.value()
            y = self.y_spinbox.value()
            z = self.z_spinbox.value()
            x4z = self.x4z_spinbox.value()
            solution_idx = self.solution_spinbox.value()
            
            self.log(f"Moving to position: ({x:.2f}, {y:.2f}, {z:.2f}) mm, x4z={x4z:.2f}, solution={solution_idx}")
            
            if self.robot.move_to_position(x, y, z, x4z_desired=x4z, solution_index=solution_idx, wait=wait):
                self.log("Cartesian movement command sent successfully.")
                # Update joint spinboxes
                self.read_joint_angles()
            else:
                self.log("Failed to send Cartesian movement command (position may be unreachable).")
                
        except Exception as e:
            self.log(f"Error moving to Cartesian position: {str(e)}")
            QMessageBox.critical(self, "Movement Error", str(e))
            
    def compute_ik_preview(self):
        """Compute IK without moving the robot."""
        try:
            x = self.x_spinbox.value()
            y = self.y_spinbox.value()
            z = self.z_spinbox.value()
            x4z = self.x4z_spinbox.value()
            
            self.log(f"Computing IK for position: ({x:.2f}, {y:.2f}, {z:.2f}) mm, x4z={x4z:.2f}")
            
            # Use existing robot or create temporary one for IK calculation
            robot = self.robot if self.robot else RobotController()
            solutions = robot.inverse_kinematics(x, y, z, x4z_desired=x4z)
            
            if solutions:
                self.log(f"Found {len(solutions)} IK solution(s):")
                for i, sol in enumerate(solutions):
                    self.log(f"  Solution {i}: {[f'{np.degrees(a):.2f}°' for a in sol]}")
            else:
                self.log("No IK solutions found (position unreachable).")
                
        except Exception as e:
            self.log(f"Error computing IK: {str(e)}")
            
    def compute_forward_kinematics(self):
        """Compute forward kinematics to get current end-effector position."""
        if not self.robot or not self.robot.is_connected:
            self.log("Robot not connected!")
            return
            
        try:
            pos, rot = self.robot.forward_kinematics()
            if pos is not None:
                self.log(f"Current end-effector position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) mm")
                # Update spinboxes
                self.x_spinbox.setValue(pos[0])
                self.y_spinbox.setValue(pos[1])
                self.z_spinbox.setValue(pos[2])
            else:
                self.log("Failed to compute forward kinematics.")
                
        except Exception as e:
            self.log(f"Error computing forward kinematics: {str(e)}")
            
    def update_status(self):
        """Update status display."""
        if not self.robot:
            self.status_display.setText("Robot not connected.")
            return
            
        try:
            status_text = "="*50 + "\n"
            status_text += "Robot Controller Status\n"
            status_text += "="*50 + "\n"
            status_text += f"Port: {self.robot.port_name}\n"
            status_text += f"Baudrate: {self.robot.baudrate}\n"
            status_text += f"Motor IDs: {self.robot.motor_ids}\n"
            status_text += f"Protocol: {self.robot.protocol_version}\n"
            status_text += f"Connected: {self.robot.is_connected}\n"
            status_text += f"Initialized: {self.robot.is_initialized}\n"
            
            if self.robot.is_connected:
                angles = self.robot.get_joint_positions()
                if angles is not None:
                    status_text += "\nCurrent Joint Angles:\n"
                    for i, (motor_id, angle) in enumerate(zip(self.robot.motor_ids, angles)):
                        status_text += f"  Joint {i+1} (Motor {motor_id}): {angle:.4f} rad ({np.degrees(angle):.2f}°)\n"
                    
                    # Calculate x4z component
                    q1, q2, q3, q4 = angles
                    x4z = np.sin(q2 + q3 + q4)
                    status_text += f"\nx4z component: {x4z:.4f}\n"
                    
                    # Show Cartesian position
                    pos_xyz, rot = self.robot.forward_kinematics()
                    if pos_xyz is not None:
                        status_text += f"\nEnd-Effector Position: [{pos_xyz[0]:.2f}, {pos_xyz[1]:.2f}, {pos_xyz[2]:.2f}] mm\n"
            
            status_text += "="*50
            self.status_display.setText(status_text)
            self.log("Status updated.")
            
        except Exception as e:
            self.log(f"Error updating status: {str(e)}")
            
    def enable_torque(self):
        """Enable torque for all motors."""
        if not self.robot or not self.robot.is_connected:
            self.log("Robot not connected!")
            return
        
        try:
            self.robot.enable_torque()
            self.log("Torque enabled for all motors.")
        except Exception as e:
            self.log(f"Error enabling torque: {str(e)}")
            
    def disable_torque(self):
        """Disable torque for all motors."""
        if not self.robot or not self.robot.is_connected:
            self.log("Robot not connected!")
            return
        
        try:
            self.robot.disable_torque()
            self.log("Torque disabled for all motors.")
        except Exception as e:
            self.log(f"Error disabling torque: {str(e)}")
            
    def set_speed(self):
        """Set motor speed."""
        if not self.robot or not self.robot.is_connected:
            self.log("Robot not connected!")
            return
        
        try:
            speed = self.speed_spinbox.value()
            self.robot.set_speed(speed)
            self.log(f"Speed set to {speed} for all motors.")
        except Exception as e:
            self.log(f"Error setting speed: {str(e)}")
            
    def detect_cameras(self):
        """Detect available camera devices."""
        self.log("Detecting available cameras...")
        available = []
        
        # Test camera indices 0-10
        for i in range(11):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        
        if available:
            self.log(f"Found {len(available)} camera(s): {available}")
            # Set to first available camera
            self.camera_device_spinbox.setValue(available[0])
            QMessageBox.information(self, "Cameras Detected", 
                f"Found camera devices at indices: {available}\n\nCamera device selector updated to: {available[0]}")
        else:
            self.log("No cameras detected")
            QMessageBox.warning(self, "No Cameras", "No camera devices were detected.")
    
    def open_camera(self):
        """Open the camera for preview."""
        try:
            device_id = self.camera_device_spinbox.value()
            # Try as string first (some systems need this), then as int
            self.camera = cv2.VideoCapture(str(device_id))
            if not self.camera.isOpened():
                # Fallback to integer if string didn't work
                self.camera = cv2.VideoCapture(device_id)
            
            if not self.camera.isOpened():
                self.log(f"Failed to open camera (device {device_id})")
                QMessageBox.warning(self, "Camera Error", 
                    f"Failed to open camera device {device_id}.\n\nTry clicking 'Detect Cameras' to find available devices.")
                self.camera = None
                return
            
            self.camera_active = True
            self.camera_open_btn.setEnabled(False)
            self.camera_close_btn.setEnabled(True)
            self.capture_btn.setEnabled(True)
            self.detection_toggle_btn.setEnabled(True)
            self.camera_timer.start(30)  # Update every 30ms (~33 FPS)
            self.log("Camera opened successfully")
            
        except Exception as e:
            self.log(f"Error opening camera: {str(e)}")
            QMessageBox.critical(self, "Camera Error", str(e))
    
    def close_camera(self):
        """Close the camera."""
        try:
            self.camera_active = False
            self.camera_timer.stop()
            
            if self.camera:
                self.camera.release()
                self.camera = None
            
            self.camera_preview_label.setText("Camera not active")
            self.camera_open_btn.setEnabled(True)
            self.camera_close_btn.setEnabled(False)
            self.capture_btn.setEnabled(False)
            self.detection_enabled = False
            self.detection_toggle_btn.setEnabled(False)
            self.detection_toggle_btn.setText("Enable Detection")
            self.log("Camera closed")
            
        except Exception as e:
            self.log(f"Error closing camera: {str(e)}")
    
    def update_camera_frame(self):
        """Update the camera preview frame."""
        if not self.camera_active or not self.camera:
            return
        
        try:
            ret, frame = self.camera.read()
            if ret:
                # Run detection if enabled
                if self.detection_enabled:
                    # Update camera height from robot FK if robot is connected
                    if self.robot and self.robot.is_connected:
                        pos, rot = self.robot.get_camera_position()
                        if pos is not None:
                            camera_height = pos[2]  # Z coordinate
                            self.detector.set_camera_height(camera_height)
                    
                    detections, mask = self.detector.detect_objects_with_positions(frame, use_hsv=True)
                    
                    # Store detections
                    self.current_detections = detections
                    self.update_detection_list()
                    
                    # Draw detections on frame
                    for i, (x, y, r, world_coords) in enumerate(detections, start=1):
                        # Draw circle and center point
                        cv2.circle(frame, (int(x), int(y)), int(r), (0, 255, 0), 2)
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                        
                        # Display pixel coordinates
                        text = f"#{i} ({int(x)},{int(y)})"
                        cv2.putText(frame, text, (int(x)+6, int(y)-6), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                        
                        # Display world coordinates if available
                        if world_coords is not None:
                            wx, wy = world_coords
                            text_world = f"World: ({wx:.1f}, {wy:.1f})"
                            cv2.putText(frame, text_world, (int(x)+6, int(y)+12), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                    
                    # Add detection count
                    count_text = f"Detections: {len(detections)}"
                    cv2.putText(frame, count_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Convert frame to Qt format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                
                # Scale to fit the label while maintaining aspect ratio
                pixmap = QPixmap.fromImage(qt_image)
                scaled_pixmap = pixmap.scaled(
                    self.camera_preview_label.width(),
                    self.camera_preview_label.height(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.camera_preview_label.setPixmap(scaled_pixmap)
            
        except Exception as e:
            self.log(f"Error updating camera frame: {str(e)}")
    
    def take_photo(self):
        """Capture and save a photo from the camera."""
        if not self.camera_active or not self.camera:
            self.log("Camera not active")
            return
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.log("Failed to capture frame")
                QMessageBox.warning(self, "Capture Error", "Failed to capture frame from camera")
                return
            
            # Create save directory if it doesn't exist
            os.makedirs(self.save_location, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_location, f"photo_{timestamp}.jpg")
            
            # Save the image
            cv2.imwrite(filename, frame)
            self.log(f"Photo saved: {filename}")
            QMessageBox.information(self, "Photo Saved", f"Photo saved successfully:\n{filename}")
            
        except Exception as e:
            self.log(f"Error taking photo: {str(e)}")
            QMessageBox.critical(self, "Capture Error", str(e))
    
    def change_save_location(self):
        """Change the directory where photos are saved."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Photo Save Location",
            self.save_location,
            QFileDialog.ShowDirsOnly
        )
        
        if directory:
            self.save_location = directory
            self.save_location_label.setText(f"Save to: {directory}")
            self.log(f"Photo save location changed to: {directory}")
    
    def toggle_detection(self):
        """Toggle object detection on/off."""
        self.detection_enabled = not self.detection_enabled
        
        if self.detection_enabled:
            self.detection_toggle_btn.setText("Disable Detection")
            self.detection_toggle_btn.setStyleSheet("background-color: #90EE90;")
            self.log("Object detection ENABLED - Camera height will be automatically updated from robot position")
        else:
            self.detection_toggle_btn.setText("Enable Detection")
            self.detection_toggle_btn.setStyleSheet("")
            self.log("Object detection DISABLED")
    
    def update_detection_list(self):
        """Update the detection list display."""
        if not self.current_detections:
            self.detection_list.setText("No detections")
            return
        
        text = f"Total Detections: {len(self.current_detections)}\n"
        text += "="*60 + "\n"
        text += f"{'#':<3} {'Pixel X':<8} {'Pixel Y':<8} {'Radius':<8} {'World X':<10} {'World Y':<10}\n"
        text += "-"*60 + "\n"
        
        for i, (x, y, r, world_coords) in enumerate(self.current_detections, start=1):
            if world_coords is not None:
                wx, wy = world_coords
                text += f"{i:<3} {int(x):<8} {int(y):<8} {int(r):<8} {wx:<10.1f} {wy:<10.1f}\n"
            else:
                text += f"{i:<3} {int(x):<8} {int(y):<8} {int(r):<8} {'N/A':<10} {'N/A':<10}\n"
        
        self.detection_list.setText(text)
        self.detection_index_spinbox.setRange(1, len(self.current_detections))
    
    def move_to_detection(self):
        """Move the robot stylus to a detected object position."""
        if not self.robot or not self.robot.is_initialized:
            self.log("Robot not connected or initialized!")
            QMessageBox.warning(self, "Not Connected", "Please connect and initialize the robot first.")
            return
        
        if not self.current_detections:
            self.log("No detections available!")
            QMessageBox.warning(self, "No Detections", "No objects detected. Enable detection first.")
            return
        
        detection_idx = self.detection_index_spinbox.value() - 1  # Convert to 0-based index
        
        if detection_idx >= len(self.current_detections):
            self.log(f"Detection #{detection_idx + 1} not available!")
            return
        
        try:
            x_pixel, y_pixel, r, world_coords = self.current_detections[detection_idx]
            print(f"World coords from detection: {world_coords}")
            if world_coords is None:
                self.log("World coordinates not available for this detection!")
                QMessageBox.warning(self, "No Coordinates", 
                    "World coordinates not available. Make sure camera height is set.")
                return
            
            wx, wy = world_coords
            
            # Step 1: Get current camera position and orientation from robot FK
            camera_pos, camera_rot = self.robot.get_camera_position()
            print(f"Camera position: {camera_pos}, rotation:\n{camera_rot}")
            if camera_pos is None:
                self.log("Failed to get camera position from robot!")
                return
            
            camera_height = camera_pos[2]  # Z coordinate of camera in base frame
            
            self.log(f"Detection #{detection_idx + 1}:")
            self.log(f"  Pixel coords: ({int(x_pixel)}, {int(y_pixel)})")
            self.log(f"  Camera frame coords from pinhole: ({wx:.1f}, {wy:.1f}) mm")
            self.log(f"  Camera position in base: ({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f}) mm")
            self.log(f"  Camera rotation matrix:\n{camera_rot}")
            
            # Step 2: Build 3D point in camera frame
            # Due to the camera's physical mounting orientation, we need to swap coordinates
            # The rotation matrix shows the camera frame is rotated relative to the stylus frame
            camera_coords = np.array([camera_height, wx, wy])
            
            print(f"Camera frame coords: {camera_coords}")
            self.log(f"  Object in camera frame: ({camera_height:.1f}, {wy:.1f}, {wx:.1f}) mm")
            
            # Step 3: Transform from camera frame to base frame using the kinematic chain
            base_coords = self.robot.camera_to_base_frame(camera_coords)
            print(f"Base frame coords: {base_coords}")
            if base_coords is None:
                self.log("Failed to transform coordinates to base frame!")
                return
            
            self.log(f"  Object in base frame (raw): ({base_coords[0]:.1f}, {base_coords[1]:.1f}, {base_coords[2]:.1f}) mm")
            
            # Step 4: The object should be on the table (Z=0 in base frame for table surface)
            # But base_coords[2] might not be 0 due to transformation issues
            # Let's set Z=0 for table surface, then add 20mm hover offset
            table_z = 0.0
            target_z = table_z + 20.0  # 20mm above table
            
            # Use XY from transformation, but override Z to be just above table
            base_coords[2] = target_z
            
            self.log(f"  Target in base frame (20mm above table): ({base_coords[0]:.1f}, {base_coords[1]:.1f}, {base_coords[2]:.1f}) mm")
            
            # Step 5: Move robot to target position using IK
            if self.robot.move_to_position(base_coords[0], base_coords[1], base_coords[2], 
                                          x4z_desired=-1.0, solution_index=0, wait=False):
                self.log(f"Successfully commanded move to detection #{detection_idx + 1}")
            else:
                self.log(f"Failed to move to detection #{detection_idx + 1} (position may be unreachable)")
                QMessageBox.warning(self, "Move Failed", 
                    "Failed to move to detection. Position may be out of reach.")
                
        except Exception as e:
            self.log(f"Error moving to detection: {str(e)}")
            QMessageBox.critical(self, "Movement Error", str(e))
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Close camera if active
        if self.camera_active:
            self.close_camera()
        
        if self.robot and self.robot.is_connected:
            reply = QMessageBox.question(
                self, 'Confirm Exit',
                'Robot is still connected. Disconnect before exiting?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.disconnect_robot()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    gui = RobotGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

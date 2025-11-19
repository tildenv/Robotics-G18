#!/usr/bin/env python3
"""
Robot Controller GUI
A PyQt5-based graphical interface for controlling the robot arm.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QTextEdit, QDoubleSpinBox,
    QSpinBox, QTabWidget, QGridLayout, QMessageBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from robot_control.robot_controller import RobotController
import math


class RobotGUI(QMainWindow):
    """Main GUI window for robot control."""
    
    def __init__(self):
        super().__init__()
        self.robot = None
        self.realtime_update_enabled = False
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
        joint_group = QGroupBox("Joint Angles (radians)")
        joint_layout = QGridLayout()
        
        self.joint_spinboxes = []
        for i in range(4):
            label = QLabel(f"Joint {i+1}:")
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-np.pi, np.pi)
            spinbox.setSingleStep(0.1)
            spinbox.setDecimals(4)
            spinbox.setValue(0.0)
            self.joint_spinboxes.append(spinbox)
            
            joint_layout.addWidget(label, i, 0)
            joint_layout.addWidget(spinbox, i, 1)
            
            # Add degree display
            deg_label = QLabel("0.00°")
            spinbox.valueChanged.connect(
                lambda val, lbl=deg_label: lbl.setText(f"{np.degrees(val):.2f}°")
            )
            joint_layout.addWidget(deg_label, i, 2)
        
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
                if self.robot.initialize(compliance_margin=0, compliance_slope=32, moving_speed=100):
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
            angles = [spinbox.value() for spinbox in self.joint_spinboxes]
            self.log(f"Moving to joint angles: {[f'{np.degrees(a):.2f}°' for a in angles]}")
            
            if self.robot.set_joint_positions(angles, wait=wait):
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
            angles = self.robot.get_joint_positions()
            if angles is not None:
                for i, (spinbox, angle) in enumerate(zip(self.joint_spinboxes, angles)):
                    spinbox.setValue(angle)
                self.log(f"Read joint angles: {[f'{np.degrees(a):.2f}°' for a in angles]}")
            else:
                self.log("Failed to read joint angles.")
                
        except Exception as e:
            self.log(f"Error reading joint angles: {str(e)}")
            
    def set_preset(self, angles):
        """Set spinboxes to preset angles."""
        for spinbox, angle in zip(self.joint_spinboxes, angles):
            spinbox.setValue(angle)
        self.log(f"Set preset angles: {[f'{np.degrees(a):.2f}°' for a in angles]}")
        
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
            
    def closeEvent(self, event):
        """Handle window close event."""
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

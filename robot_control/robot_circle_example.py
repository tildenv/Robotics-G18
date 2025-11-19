#!/usr/bin/env python3
"""
Smooth Circular Trajectory Example
Uses quintic polynomial interpolation to generate smooth robot motion along a circular path.
Based on the trajectory planning from IK_fixed.ipynb
"""

import numpy as np
import time
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robot_control.robot_controller import RobotController


def forward_kin_numeric(q, robot):
    """
    Forward kinematics using numpy (from notebook).
    Returns list of transformation matrices [T0, T1, T2, T3, T4].
    
    Args:
        q: Joint angles [q1, q2, q3, q4] in radians
        robot: RobotController instance (to access DH parameters)
    """
    # DH parameters: theta, d, a, alpha
    # A1 = DH(q[0], L1, 0, pi/2)
    # A2 = DH(q[1], 0, L2, 0)
    # A3 = DH(q[2], 0, L3, 0)
    # A4 = DH(q[3], 0, L4, 0)
    
    q1, q2, q3, q4 = q
    L1, L2, L3, L4 = robot.L1, robot.L2, robot.L3, robot.L4
    
    # A1: DH(q1, L1, 0, pi/2)
    c1, s1 = np.cos(q1), np.sin(q1)
    A1 = np.array([
        [c1, 0, s1, 0],
        [s1, 0, -c1, 0],
        [0, 1, 0, L1],
        [0, 0, 0, 1]
    ])
    
    # A2: DH(q2, 0, L2, 0)
    c2, s2 = np.cos(q2), np.sin(q2)
    A2 = np.array([
        [c2, -s2, 0, L2*c2],
        [s2, c2, 0, L2*s2],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # A3: DH(q3, 0, L3, 0)
    c3, s3 = np.cos(q3), np.sin(q3)
    A3 = np.array([
        [c3, -s3, 0, L3*c3],
        [s3, c3, 0, L3*s3],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # A4: DH(q4, 0, L4, 0)
    c4, s4 = np.cos(q4), np.sin(q4)
    A4 = np.array([
        [c4, -s4, 0, L4*c4],
        [s4, c4, 0, L4*s4],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    T0 = np.eye(4)
    T1 = A1
    T2 = T1 @ A2
    T3 = T2 @ A3
    T4 = T3 @ A4
    
    return [T0, T1, T2, T3, T4]


def compute_geometric_jacobian(T_list, frame_index=4):
    """
    Compute geometric Jacobian exactly as in the notebook.
    
    Args:
        T_list: List of transformation matrices from forward kinematics
        frame_index: Which frame to compute Jacobian for (default: 4 for end-effector)
        
    Returns:
        J: 6x4 geometric Jacobian matrix
    """
    o_n = np.array(T_list[frame_index][:3, 3], dtype=float)
    J = np.zeros((6, 4), dtype=float)
    
    for i in range(1, 5):  # joints 1..4
        o_im1 = np.array(T_list[i-1][:3, 3], dtype=float)   # origin of frame {i-1}
        z_im1 = np.array(T_list[i-1][:3, 2], dtype=float)   # z-axis of frame {i-1} in base
        
        J[:3, i-1] = np.cross(z_im1, (o_n - o_im1))         # linear velocity
        J[3:, i-1] = z_im1                                  # angular velocity
    
    return J


def calculate_joint_velocity(robot, q, v_task_linear):
    """
    Calculate joint velocities for a given task-space linear velocity.
    This is exactly as implemented in the notebook (Problem 5).
    
    Args:
        robot: RobotController instance (for DH parameters)
        q: Joint configuration in radians
        v_task_linear: Desired linear velocity [vx, vy, vz]
        
    Returns:
        q_dot: Joint velocities
    """
    # Get geometric Jacobian using forward kinematics
    T_list = forward_kin_numeric(q, robot)
    J_geom = compute_geometric_jacobian(T_list, frame_index=4)
    T4 = T_list[4]
    
    # Extract components
    J_v = J_geom[0:3, :]  # Linear velocity Jacobian (3x4)
    J_w = J_geom[3:6, :]  # Angular velocity Jacobian (3x4)
    x4_vec = T4[0:3, 0]   # x-axis of frame 4
    
    x4_x = x4_vec[0]
    x4_y = x4_vec[1]
    
    # Construct the 4th row of the Task Jacobian
    # This ensures rotation around z-axis is constrained
    J_task_row4 = (J_w[0, :] * x4_y) - (J_w[1, :] * x4_x)
    
    # Stack to create 4x4 task Jacobian
    J_task = np.vstack([J_v, J_task_row4])
    
    # Task-space velocity (3 linear + 1 rotational constraint)
    f_dot = np.array([v_task_linear[0], v_task_linear[1], v_task_linear[2], 0.0])
    
    # Solve for joint velocities
    try:
        q_dot = np.linalg.solve(J_task, f_dot)
    except np.linalg.LinAlgError:
        print(f"Warning: Jacobian is singular at q = {q}. Using pseudo-inverse.")
        q_dot = np.linalg.pinv(J_task) @ f_dot
    
    return q_dot


def solve_quintic_coeffs(t_start, t_end, y_constraints):
    """
    Solve for quintic polynomial coefficients given boundary conditions.
    
    Args:
        t_start: Start time
        t_end: End time
        y_constraints: [q(t_start), qd(t_start), qdd(t_start), q(t_end), qd(t_end), qdd(t_end)]
        
    Returns:
        Coefficients [c0, c1, c2, c3, c4, c5] for polynomial q(t) = c0 + c1*t + ... + c5*t^5
    """
    tA = t_start
    tB = t_end
    
    # The M matrix from standard quintic polynomial formulation
    M = np.array([
        [1, tA, tA**2, tA**3, tA**4, tA**5],
        [0, 1, 2*tA, 3*tA**2, 4*tA**3, 5*tA**4],
        [0, 0, 2, 6*tA, 12*tA**2, 20*tA**3],
        [1, tB, tB**2, tB**3, tB**4, tB**5],
        [0, 1, 2*tB, 3*tB**2, 4*tB**3, 5*tB**4],
        [0, 0, 2, 6*tB, 12*tB**2, 20*tB**3]
    ])
    
    # Solve the system M * c = y
    c = np.linalg.solve(M, y_constraints)
    return c


def eval_quintic(coeffs, t):
    """
    Evaluate a quintic polynomial at time t.
    
    Args:
        coeffs: [c0, c1, c2, c3, c4, c5]
        t: Time value
        
    Returns:
        Polynomial value at time t
    """
    return coeffs[0] + coeffs[1]*t + coeffs[2]*t**2 + coeffs[3]*t**3 + coeffs[4]*t**4 + coeffs[5]*t**5


def generate_circular_trajectory(robot, center, radius, num_points=37):
    """
    Generate IK solutions for points along a circular path.
    
    Args:
        robot: RobotController instance
        center: Center point [x, y, z] in mm
        radius: Circle radius in mm
        num_points: Number of points around the circle
        
    Returns:
        List of joint configurations (in radians)
    """
    print(f"Generating circular trajectory: center={center}, radius={radius}mm, points={num_points}")
    
    configs = []
    for j in range(num_points):
        phi = 2 * np.pi * j / (num_points - 1)
        
        # Calculate point on circle in YZ plane
        x = center[0]
        y = center[1] + radius * np.cos(phi)
        z = center[2] + radius * np.sin(phi)
        
        # Compute IK solutions
        solutions = robot.inverse_kinematics(x, y, z, x4z_desired=0.0)
        
        if not solutions:
            print(f"Warning: Point ({x:.1f}, {y:.1f}, {z:.1f}) is unreachable!")
            continue
        
        # Use first solution
        configs.append(solutions[0])
    
    print(f"Generated {len(configs)} configurations")
    return configs


def generate_smooth_trajectory(configs, segment_duration=2.0):
    """
    Generate smooth trajectory using quintic polynomials between keypoints.
    
    Args:
        configs: List of joint configurations (in radians)
        segment_duration: Duration for each segment in seconds
        
    Returns:
        Dictionary of polynomial coefficients for each segment and joint
    """
    # Select 5 keypoints (start, 1/4, 1/2, 3/4, end)
    num_configs = len(configs)
    keypoint_indices = [0, num_configs//4, num_configs//2, 3*num_configs//4, num_configs-1]
    
    q_knots = [configs[i] for i in keypoint_indices]
    
    # Define Cartesian velocities at knots (from Problem 6 specification)
    # These ensure smooth continuous motion through the circular path
    v_cartesian_knots = [
        np.array([0.0, 0.0, 0.0]),      # Start (Rest)
        np.array([0.0, -27.0, 0.0]),    # Knot 1 (Moving down Y)
        np.array([0.0, 0.0, -27.0]),    # Knot 2 (Moving down Z)
        np.array([0.0, 27.0, 0.0]),     # Knot 3 (Moving up Y)
        np.array([0.0, 0.0, 0.0])       # End (Rest)
    ]
    
    # Calculate joint velocities using Inverse Jacobian
    # This converts the desired Cartesian velocities to joint space
    # Note: We need a robot instance for DH parameters
    # Create a temporary controller just for kinematics (no connection needed)
    from robot_control.robot_controller import RobotController
    temp_robot = RobotController()
    
    qd_knots = []
    for q, v in zip(q_knots, v_cartesian_knots):
        q_dot = calculate_joint_velocity(temp_robot, q, v)
        qd_knots.append(q_dot)
    
    # Zero accelerations at all knots
    qdd_knots = [
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0])
    ]
    
    knot_states = list(zip(q_knots, qd_knots, qdd_knots))
    
    # Compute polynomial coefficients for each segment
    segment_names = ['A', 'B', 'C', 'D']
    all_coeffs = {}
    t_start = 0.0
    t_end = segment_duration
    
    print("\nComputing quintic polynomial coefficients...")
    for i in range(len(segment_names)):
        seg_name = segment_names[i]
        all_coeffs[seg_name] = []
        
        start_state = knot_states[i]
        end_state = knot_states[i + 1]
        
        for j in range(4):  # For each joint
            y_vec = [
                start_state[0][j],  # q(start)
                start_state[1][j],  # qd(start)
                start_state[2][j],  # qdd(start)
                end_state[0][j],    # q(end)
                end_state[1][j],    # qd(end)
                end_state[2][j]     # qdd(end)
            ]
            
            c = solve_quintic_coeffs(t_start, t_end, y_vec)
            all_coeffs[seg_name].append(c)
        
        print(f"  Segment {seg_name}: coefficients computed")
    
    return all_coeffs


def generate_trajectory_from_coeffs(all_coeffs, segment_duration=2.0, dt=0.04):
    """
    Generate trajectory points from polynomial coefficients.
    
    Args:
        all_coeffs: Dictionary of polynomial coefficients
        segment_duration: Duration of each segment in seconds
        dt: Time step in seconds
        
    Returns:
        List of joint configurations (in radians)
    """
    q_trajectory = []
    segment_names = ['A', 'B', 'C', 'D']
    total_duration = segment_duration * len(segment_names)
    
    # Create array of time points
    t_global_array = np.arange(0, total_duration + dt, dt)
    
    for t_global in t_global_array:
        if t_global > total_duration:
            t_global = total_duration
        
        # Find the correct segment
        seg_index = int(t_global // segment_duration)
        if seg_index >= len(segment_names):
            seg_index = len(segment_names) - 1
        
        seg_name = segment_names[seg_index]
        
        # Calculate local time for segment
        t_local = t_global - (seg_index * segment_duration)
        if t_global == total_duration:
            t_local = segment_duration
        
        # Calculate q vector for this time
        q_vec = np.zeros(4)
        for j in range(4):  # For each joint
            coeffs = all_coeffs[seg_name][j]
            q_vec[j] = eval_quintic(coeffs, t_local)
        
        q_trajectory.append(q_vec)
    
    return q_trajectory


def execute_trajectory(robot, trajectory, dt=0.04, preview=True):
    """
    Execute a trajectory on the robot.
    
    Args:
        robot: RobotController instance
        trajectory: List of joint configurations in radians
        dt: Time step in seconds
        preview: If True, show trajectory summary before executing
        
    Returns:
        List of actual end-effector positions during execution
    """
    print(f"\n{'='*60}")
    print(f"Trajectory Execution")
    print(f"{'='*60}")
    print(f"Number of waypoints: {len(trajectory)}")
    print(f"Time step: {dt}s")
    print(f"Total duration: {len(trajectory) * dt:.2f}s")
    
    if preview:
        print(f"\nFirst waypoint (radians): {np.round(trajectory[0], 4)}")
        print(f"First waypoint (degrees): {np.round(np.degrees(trajectory[0]), 2)}")
        
        response = input("\nReady to execute? (y/n): ")
        if response.lower() != 'y':
            print("Execution cancelled")
            return None
    
    print("\nExecuting trajectory...")
    start_time = time.time()
    
    # Track actual positions
    actual_positions = []
    
    for i, q_radians in enumerate(trajectory):
        # Send positions to motors (now accepts radians directly)
        robot.set_joint_positions(list(q_radians), wait=False)
        
        # Record ACTUAL end-effector position from current robot state
        current_q = robot.get_joint_positions()
        if current_q is not None:
            pos, _ = robot.forward_kinematics(current_q)
            if pos is not None:
                actual_positions.append(pos.copy())
        
        # Progress indicator
        if i % 25 == 0:
            elapsed = time.time() - start_time
            progress = 100.0 * i / len(trajectory)
            print(f"  Progress: {progress:.1f}% | Waypoint {i}/{len(trajectory)} | Elapsed: {elapsed:.2f}s")
        
        # Wait for next time step
        time.sleep(dt)
    
    total_time = time.time() - start_time
    print(f"\nTrajectory complete! Total time: {total_time:.2f}s")
    print(f"Recorded {len(actual_positions)} positions")
    return actual_positions


def plot_trajectory(trajectory, actual_positions, center, radius, robot):
    """
    Plot the planned trajectory and actual executed trajectory.
    
    Args:
        trajectory: List of joint configurations (planned)
        actual_positions: List of actual end-effector positions
        center: Center of the circular path
        radius: Radius of the circular path
        robot: RobotController instance (for DH parameters)
    """
    print("\nGenerating trajectory plot...")
    
    # Compute planned end-effector positions from joint angles
    planned_positions = []
    for q in trajectory:
        T_list = forward_kin_numeric(q, robot)
        pos = T_list[4][:3, 3]
        planned_positions.append(pos)
    
    planned_positions = np.array(planned_positions)
    
    # Generate ideal circle for reference
    phi_ideal = np.linspace(0, 2*np.pi, 200)
    ideal_x = center[0] + np.zeros_like(phi_ideal)
    ideal_y = center[1] + radius * np.cos(phi_ideal)
    ideal_z = center[2] + radius * np.sin(phi_ideal)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Plot ideal circle
    ax1.plot(ideal_x, ideal_y, ideal_z, 'r--', linewidth=2, label='Ideal Circle', alpha=0.7)
    
    # Plot planned trajectory
    ax1.plot(planned_positions[:, 0], planned_positions[:, 1], planned_positions[:, 2], 
             'b-', linewidth=2, label='Planned Trajectory', alpha=0.8)
    
    # Plot actual trajectory if available
    if actual_positions:
        actual_pos_array = np.array(actual_positions)
        ax1.plot(actual_pos_array[:, 0], actual_pos_array[:, 1], actual_pos_array[:, 2], 
                 'g-', linewidth=1.5, label='Actual Trajectory', alpha=0.9)
        
        # Mark start and end points
        ax1.scatter(*actual_pos_array[0], color='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(*actual_pos_array[-1], color='red', s=100, marker='x', label='End', zorder=5)
    
    # Mark center
    ax1.scatter(*center, color='black', s=50, marker='+', label='Center')
    
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = radius * 3
    ax1.set_xlim(center[0] - max_range, center[0] + max_range)
    ax1.set_ylim(center[1] - max_range, center[1] + max_range)
    ax1.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # XY plane view
    ax2 = fig.add_subplot(222)
    ax2.plot(ideal_y, ideal_x, 'r--', linewidth=2, label='Ideal', alpha=0.7)
    ax2.plot(planned_positions[:, 1], planned_positions[:, 0], 'b-', linewidth=2, label='Planned', alpha=0.8)
    if actual_positions:
        actual_pos_array = np.array(actual_positions)
        ax2.plot(actual_pos_array[:, 1], actual_pos_array[:, 0], 'g-', linewidth=1.5, label='Actual', alpha=0.9)
        ax2.scatter(actual_pos_array[0, 1], actual_pos_array[0, 0], color='green', s=100, marker='o', zorder=5)
        ax2.scatter(actual_pos_array[-1, 1], actual_pos_array[-1, 0], color='red', s=100, marker='x', zorder=5)
    ax2.scatter(center[1], center[0], color='black', s=50, marker='+')
    ax2.set_xlabel('Y (mm)')
    ax2.set_ylabel('X (mm)')
    ax2.set_title('XY Plane View')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # YZ plane view (circle plane)
    ax3 = fig.add_subplot(223)
    ax3.plot(ideal_y, ideal_z, 'r--', linewidth=2, label='Ideal', alpha=0.7)
    ax3.plot(planned_positions[:, 1], planned_positions[:, 2], 'b-', linewidth=2, label='Planned', alpha=0.8)
    if actual_positions:
        actual_pos_array = np.array(actual_positions)
        ax3.plot(actual_pos_array[:, 1], actual_pos_array[:, 2], 'g-', linewidth=1.5, label='Actual', alpha=0.9)
        ax3.scatter(actual_pos_array[0, 1], actual_pos_array[0, 2], color='green', s=100, marker='o', zorder=5)
        ax3.scatter(actual_pos_array[-1, 1], actual_pos_array[-1, 2], color='red', s=100, marker='x', zorder=5)
    ax3.scatter(center[1], center[2], color='black', s=50, marker='+')
    ax3.set_xlabel('Y (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('YZ Plane View (Circle Plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Error plot if actual positions available
    ax4 = fig.add_subplot(224)
    if actual_positions:
        actual_pos_array = np.array(actual_positions)
        # Calculate error from ideal circle
        errors = []
        for pos in actual_pos_array:
            # Distance from center in YZ plane
            dy = pos[1] - center[1]
            dz = pos[2] - center[2]
            dist = np.sqrt(dy**2 + dz**2)
            error = abs(dist - radius)
            errors.append(error)
        
        time_points = np.arange(len(errors)) * 0.04  # Assuming dt=0.04
        ax4.plot(time_points, errors, 'b-', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Radial Error (mm)')
        ax4.set_title('Trajectory Error from Ideal Circle')
        ax4.grid(True, alpha=0.3)
        
        # Print statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        std_error = np.std(errors)
        print(f"\nTrajectory Error Statistics:")
        print(f"  Mean error: {mean_error:.3f} mm")
        print(f"  Max error: {max_error:.3f} mm")
        print(f"  Std error: {std_error:.3f} mm")
    else:
        ax4.text(0.5, 0.5, 'No actual trajectory data', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Error Plot')
    
    plt.tight_layout()
    plt.savefig('/home/stas/dtu/robotics/project/trajectory_plot.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved to: /home/stas/dtu/robotics/project/trajectory_plot.png")
    plt.show()



def main():
    """Main execution function."""
    print("="*60)
    print("Smooth Circular Trajectory Demo")
    print("="*60)
    
    # Create robot controller
    robot = RobotController(
        port_name='/dev/ttyACM0',
        baudrate=1000000,
        motor_ids=[1, 2, 3, 4]
    )
    
    try:
        # Connect and initialize
        print("\n[1/6] Connecting to robot...")
        if not robot.connect():
            return
        
        print("\n[2/6] Initializing motors...")
        if not robot.initialize(compliance_margin=0, compliance_slope=32, moving_speed=50):
            return
        
        robot.print_status()
        
        # Define circular path parameters (from notebook)
        center = np.array([100.0, 0.0, 120.0])  # mm
        radius = 22.0  # mm
        
        # Generate circular trajectory waypoints
        print("\n[3/6] Generating circular trajectory...")
        configs = generate_circular_trajectory(robot, center, radius, num_points=37)
        
        if not configs:
            print("Failed to generate trajectory!")
            return
        
        # Generate smooth trajectory with quintic polynomials
        print("\n[4/6] Computing smooth trajectory...")
        all_coeffs = generate_smooth_trajectory(configs, segment_duration=2.0)
        
        # Generate trajectory points
        print("\n[5/6] Generating trajectory waypoints...")
        dt = 0.04  # 25 Hz update rate
        segment_duration = 2.0
        trajectory = generate_trajectory_from_coeffs(all_coeffs, segment_duration=segment_duration, dt=dt)
        print(f"Generated {len(trajectory)} waypoints")
        
        # Execute trajectory
        print("\n[6/6] Executing trajectory...")
        actual_positions = execute_trajectory(robot, trajectory, dt=dt, preview=True)
        
        if actual_positions is not None:
            print("\n✓ Mission accomplished!")
            time.sleep(1)
            
            # Plot the trajectory
            print("\nGenerating plots...")
            plot_trajectory(trajectory, actual_positions, center, radius, robot)
            
            # Return to home
            print("\nReturning to home position...")
            robot.set_joint_positions([0.0, 0.0, 0.0, 0.0], wait=True)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\nCleaning up...")
        robot.disable_torque()
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()

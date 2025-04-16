from math import sin, cos, degrees
import numpy as np
from matplotlib.figure import Figure
import time
from helper_fcns.utils import (
    EndEffector,
    rotm_to_euler,
    dh_to_matrix,
    euler_to_rotm,
    wraptopi,
)

PI = 3.1415926535897932384
np.set_printoptions(precision=3)


class Robot:
    """
    Represents a robot manipulator with various kinematic configurations.
    Provides methods to calculate forward kinematics, inverse kinematics, and velocity kinematics.
    Also includes methods to visualize the robot's motion and state in 3D.

    Attributes:
        num_joints (int): Number of joints in the robot.
        ee_coordinates (list): List of end-effector coordinates.
        robot (object): The robot object (e.g., TwoDOFRobot, ScaraRobot, etc.).
        origin (list): Origin of the coordinate system.
        axes_length (float): Length of the axes for visualization.
        point_x, point_y, point_z (list): Lists to store coordinates of points for visualization.
        show_animation (bool): Whether to show the animation or not.
        plot_limits (list): Limits for the plot view.
        fig (matplotlib.figure.Figure): Matplotlib figure for 3D visualization.
        sub1 (matplotlib.axes._subplots.Axes3DSubplot): Matplotlib 3D subplot.
    """

    def __init__(self, type="2-dof", show_animation: bool = True):
        """
        Initializes a robot with a specific configuration based on the type.

        Args:
            type (str, optional): Type of robot (e.g., '2-dof', 'scara', '5-dof'). Defaults to '2-dof'.
            show_animation (bool, optional): Whether to show animation of robot movement. Defaults to True.
        """
        if type == "2-dof":
            self.num_joints = 2
            self.ee_coordinates = ["X", "Y"]
            self.robot = TwoDOFRobot()

        elif type == "scara":
            self.num_joints = 3
            self.ee_coordinates = ["X", "Y", "Z", "Theta"]
            self.robot = ScaraRobot()

        elif type == "5-dof":
            self.num_joints = 5
            self.ee_coordinates = ["X", "Y", "Z", "RotX", "RotY", "RotZ"]
            self.robot = FiveDOFRobot()

        self.origin = [0.0, 0.0, 0.0]
        self.axes_length = 0.04
        self.point_x, self.point_y, self.point_z = [], [], []
        self.waypoint_x, self.waypoint_y, self.waypoint_z = [], [], []
        self.waypoint_rotx, self.waypoint_roty, self.waypoint_rotz = [], [], []
        self.show_animation = show_animation
        self.plot_limits = [0.65, 0.65, 0.8]

        if self.show_animation:
            self.fig = Figure(figsize=(12, 10), dpi=100)
            self.sub1 = self.fig.add_subplot(1, 1, 1, projection="3d")
            self.fig.suptitle("Manipulator Kinematics Visualization", fontsize=16)

        # initialize figure plot
        self.init_plot()

    def init_plot(self):
        """Initializes the plot by calculating the robot's points and calling the plot function."""
        self.robot.calc_robot_points()
        self.plot_3D()

    def update_plot(self, pose=None, angles=None, soln=0, numerical=False):
        """
        Updates the robot's state based on new pose or joint angles and updates the visualization.

        Args:
            pose (EndEffector, optional): Desired end-effector pose for inverse kinematics.
            angles (list, optional): Joint angles for forward kinematics.
            soln (int, optional): The inverse kinematics solution to use (0 or 1).
            numerical (bool, optional): Whether to use numerical inverse kinematics.
        """
        if pose is not None:  # Inverse kinematics case
            if not numerical:
                self.robot.calc_inverse_kinematics(pose, soln=soln)
            else:
                self.robot.calc_numerical_ik(pose, tol=0.02, ilimit=50)
        elif angles is not None:  # Forward kinematics case
            self.robot.calc_forward_kinematics(angles, radians=False)
        else:
            return
        self.plot_3D()

    def move_velocity(self, vel):
        """
        Moves the robot based on a given velocity input.

        Args:
            vel (list): Velocity input for the robot.
        """
        self.robot.calc_velocity_kinematics(vel)
        self.plot_3D()

    def draw_line_3D(self, p1, p2, format_type: str = "k-"):
        """
        Draws a 3D line between two points.

        Args:
            p1 (list): Coordinates of the first point.
            p2 (list): Coordinates of the second point.
            format_type (str, optional): The format of the line. Defaults to "k-".
        """
        self.sub1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], format_type)

    def draw_ref_line(self, point, axes=None, ref="xyz"):
        """
        Draws reference lines from a given point along specified axes.

        Args:
            point (list): The coordinates of the point to draw from.
            axes (matplotlib.axes, optional): The axes on which to draw the reference lines.
            ref (str, optional): Which reference axes to draw ('xyz', 'xy', or 'xz'). Defaults to 'xyz'.
        """
        line_width = 0.7
        if ref == "xyz":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[1], point[1]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]],
                [point[1], self.plot_limits[1]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # Y line
            axes.plot(
                [point[0], point[0]],
                [point[1], point[1]],
                [point[2], 0.0],
                "b--",
                linewidth=line_width,
            )  # Z line
        elif ref == "xy":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[1], point[1]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]],
                [point[1], self.plot_limits[1]],
                "b--",
                linewidth=line_width,
            )  # Y line
        elif ref == "xz":
            axes.plot(
                [point[0], self.plot_limits[0]],
                [point[2], point[2]],
                "b--",
                linewidth=line_width,
            )  # X line
            axes.plot(
                [point[0], point[0]], [point[2], 0.0], "b--", linewidth=line_width
            )  # Z line

    def plot_waypoints(self):
        """
        Plots the waypoints in the 3D visualization
        """
        # draw the points
        self.sub1.plot(
            self.waypoint_x, self.waypoint_y, self.waypoint_z, "or", markersize=8
        )

    def update_waypoints(self, waypoints: list):
        """
        Updates the waypoints into a member variable
        """
        for i in range(len(waypoints)):
            self.waypoint_x.append(waypoints[i][0])
            self.waypoint_y.append(waypoints[i][1])
            self.waypoint_z.append(waypoints[i][2])
            # self.waypoint_rotx.append(waypoints[i][3])
            # self.waypoint_roty.append(waypoints[i][4])
            # self.waypoint_rotz.append(waypoints[i][5])

    def plot_3D(self):
        """
        Plots the 3D visualization of the robot, including the robot's links, end-effector, and reference frames.
        """
        self.sub1.cla()
        self.point_x.clear()
        self.point_y.clear()
        self.point_z.clear()

        EE = self.robot.ee

        # draw lines to connect the points
        for i in range(len(self.robot.points) - 1):
            self.draw_line_3D(self.robot.points[i], self.robot.points[i + 1])

        # draw the points
        for i in range(len(self.robot.points)):
            self.point_x.append(self.robot.points[i][0])
            self.point_y.append(self.robot.points[i][1])
            self.point_z.append(self.robot.points[i][2])
        self.sub1.plot(
            self.point_x,
            self.point_y,
            self.point_z,
            marker="o",
            markerfacecolor="m",
            markersize=12,
        )

        # draw the waypoints
        self.plot_waypoints()

        # draw the EE
        self.sub1.plot(EE.x, EE.y, EE.z, "bo")
        # draw the base reference frame
        self.draw_line_3D(
            self.origin,
            [self.origin[0] + self.axes_length, self.origin[1], self.origin[2]],
            format_type="r-",
        )
        self.draw_line_3D(
            self.origin,
            [self.origin[0], self.origin[1] + self.axes_length, self.origin[2]],
            format_type="g-",
        )
        self.draw_line_3D(
            self.origin,
            [self.origin[0], self.origin[1], self.origin[2] + self.axes_length],
            format_type="b-",
        )
        # draw the EE reference frame
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[0], format_type="r-")
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[1], format_type="g-")
        self.draw_line_3D([EE.x, EE.y, EE.z], self.robot.EE_axes[2], format_type="b-")
        # draw reference / trace lines
        self.draw_ref_line([EE.x, EE.y, EE.z], self.sub1, ref="xyz")

        # add text at bottom of window
        pose_text = "End-effector Pose:      [ "
        pose_text += f"X: {round(EE.x,4)},  "
        pose_text += f"Y: {round(EE.y,4)},  "
        pose_text += f"Z: {round(EE.z,4)},  "
        pose_text += f"RotX: {round(EE.rotx,4)},  "
        pose_text += f"RotY: {round(EE.roty,4)},  "
        pose_text += f"RotZ: {round(EE.rotz,4)}  "
        pose_text += " ]"

        theta_text = "Joint Positions (deg/m):     ["
        for i in range(self.num_joints):
            theta_text += f" {round(np.rad2deg(self.robot.theta[i]),2)}, "
        theta_text += " ]"

        textstr = pose_text + "\n" + theta_text
        self.sub1.text2D(
            0.2, 0.02, textstr, fontsize=13, transform=self.fig.transFigure
        )

        self.sub1.set_xlim(-self.plot_limits[0], self.plot_limits[0])
        self.sub1.set_ylim(-self.plot_limits[1], self.plot_limits[1])
        self.sub1.set_zlim(0, self.plot_limits[2])
        self.sub1.set_xlabel("x [m]")
        self.sub1.set_ylabel("y [m]")


class TwoDOFRobot:
    """
    Represents a 2-degree-of-freedom (DOF) robot arm with two joints and one end effector.
    Includes methods for calculating forward kinematics (FPK), inverse kinematics (IPK),
    and velocity kinematics (VK).

    Attributes:
        l1 (float): Length of the first arm segment.
        l2 (float): Length of the second arm segment.
        theta (list): Joint angles.
        theta_limits (list): Joint limits for each joint.
        ee (EndEffector): The end effector object.
        points (list): List of points representing the robot's configuration.
        num_dof (int): The number of degrees of freedom (2 for this robot).
    """

    def __init__(self):
        """
        Initializes a 2-DOF robot with default arm segment lengths and joint angles.
        """
        self.l1 = 0.30  # Length of the first arm segment
        self.l2 = 0.25  # Length of the second arm segment

        self.theta = [0.0, 0.0]  # Joint angles (in radians)
        self.theta_limits = [[-PI, PI], [-PI + 0.261, PI - 0.261]]  # Joint limits

        self.ee = EndEffector()  # The end-effector object
        self.num_dof = 2  # Number of degrees of freedom
        self.points = [None] * (self.num_dof + 1)  # List to store robot points

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculates the forward kinematics for the robot based on the joint angles.

        Args:
            theta (list): Joint angles.
            radians (bool, optional): Whether the angles are in radians or degrees. Defaults to False.
        """
        if not radians:
            # Convert degrees to radians if the input is in degrees
            self.theta[0] = np.deg2rad(theta[0])
            self.theta[1] = np.deg2rad(theta[1])
        else:
            self.theta = theta

        # Ensure that the joint angles respect the joint limits
        for i, th in enumerate(self.theta):
            self.theta[i] = np.clip(
                th, self.theta_limits[i][0], self.theta_limits[i][1]
            )

        # Update the robot configuration (i.e., the positions of the joints and end effector)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculates the inverse kinematics (IK) for a given end effector position.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            soln (int, optional): The solution branch to use. Defaults to 0 (first solution).
        """
        x, y = EE.x, EE.y
        l1, l2 = self.l1, self.l2
        ########################################
        L = np.sqrt(x**2 + y**2)
        beta = np.arccos((l1**2 + l2**2 - L**2) / (2 * l1 * l2))
        if soln:
            self.theta[1] = np.pi - beta
        else:
            self.theta[1] = np.pi + beta
        alpha = np.arctan2(l2 * np.sin(self.theta[1]), l1 + l2 * np.cos(self.theta[1]))
        gamma = np.arctan2(y, x)
        self.theta[0] = gamma - alpha

        ########################################
        print(f"{self.theta=}")
        # Calculate robot points based on the updated joint angles
        self.calc_robot_points()

    def calc_numerical_ik(self, EE: EndEffector, tol=0.01, ilimit=50):
        """
        Calculates numerical inverse kinematics (IK) based on input end effector coordinates.

        Args:
            EE (EndEffector): The end effector object containing the target position (x, y).
            tol (float, optional): The tolerance for the solution. Defaults to 0.01.
            ilimit (int, optional): The maximum number of iterations. Defaults to 50.
        """

        x, y = EE.x, EE.y

        ########################################

        i = 0  #
        xd = np.array([x, y])
        self.calc_forward_kinematics(self.theta)
        e = xd - np.array([self.ee.x, self.ee.y])

        while (any(abs(e)) > tol) or i >= ilimit:
            xd = np.array([x, y])
            self.calc_forward_kinematics(self.theta)
            e = xd - np.array([self.ee.x, self.ee.y])
            self.theta = self.theta + self.inverse_jacobian() * e
            i += 1

        ########################################

        self.calc_robot_points()

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculates the velocity kinematics for the robot based on the given velocity input.

        Args:
            vel (list): The velocity vector for the end effector [vx, vy].
        """

        ########################################

        # insert your code here

        ########################################

        # Update robot points based on the new joint angles
        self.calc_robot_points()

    def jacobian(self, theta: list = None):
        """
        Returns the Jacobian matrix for the robot. If theta is not provided,
        the function will use the object's internal theta attribute.

        Args:
            theta (list, optional): The joint angles for the robot. Defaults to self.theta.

        Returns:
            np.ndarray: The Jacobian matrix (2x2).
        """
        # Use default values if arguments are not provided
        if theta is None:
            theta = self.theta

        return np.array(
            [
                [
                    -self.l1 * sin(theta[0]) - self.l2 * sin(theta[0] + theta[1]),
                    -self.l2 * sin(theta[0] + theta[1]),
                ],
                [
                    self.l1 * cos(theta[0]) + self.l2 * cos(theta[0] + theta[1]),
                    self.l2 * cos(theta[0] + theta[1]),
                ],
            ]
        )

    def inverse_jacobian(self):
        """
        Returns the inverse of the Jacobian matrix.

        Returns:
            np.ndarray: The inverse Jacobian matrix.
        """
        J = self.jacobian()
        print(f"Determinant of J: {np.linalg.det(J)}")
        # return np.linalg.inv(self.jacobian())
        return np.linalg.pinv(self.jacobian())

    def calc_robot_points(self):
        """
        Calculates the positions of the robot's joints and the end effector.

        Updates the `points` list, storing the coordinates of the base, shoulder, elbow, and end effector.
        """
        # Base position
        self.points[0] = [0.0, 0.0, 0.0]
        # Shoulder joint
        self.points[1] = [
            self.l1 * cos(self.theta[0]),
            self.l1 * sin(self.theta[0]),
            0.0,
        ]
        # Elbow joint
        self.points[2] = [
            self.l1 * cos(self.theta[0]) + self.l2 * cos(self.theta[0] + self.theta[1]),
            self.l1 * sin(self.theta[0]) + self.l2 * sin(self.theta[0] + self.theta[1]),
            0.0,
        ]

        # Update end effector position
        self.ee.x = self.points[2][0]
        self.ee.y = self.points[2][1]
        self.ee.z = self.points[2][2]
        self.ee.rotz = self.theta[0] + self.theta[1]

        # End effector axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = (
            np.array(
                [
                    cos(self.theta[0] + self.theta[1]),
                    sin(self.theta[0] + self.theta[1]),
                    0,
                ]
            )
            * 0.075
            + self.points[2]
        )
        self.EE_axes[1] = (
            np.array(
                [
                    -sin(self.theta[0] + self.theta[1]),
                    cos(self.theta[0] + self.theta[1]),
                    0,
                ]
            )
            * 0.075
            + self.points[2]
        )
        self.EE_axes[2] = np.array([0, 0, 1]) * 0.075 + self.points[2]


class ScaraRobot:
    """
    A class representing a SCARA (Selective Compliance Assembly Robot Arm) robot.
    This class handles the kinematics (forward, inverse, and velocity kinematics)
    and robot configuration, including joint limits and end-effector calculations.
    """

    def __init__(self):
        """
        Initializes the SCARA robot with its geometry, joint variables, and limits.
        Sets up the transformation matrices and robot points.
        """
        # Geometry of the robot (link lengths in meters)
        self.l1 = 0.35  # Base to 1st joint
        self.l2 = 0.18  # 1st joint to 2nd joint
        self.l3 = 0.15  # 2nd joint to 3rd joint
        self.l4 = 0.30  # 3rd joint to 4th joint (tool or end-effector)
        self.l5 = 0.12  # Tool offset

        # Joint variables (angles in radians)
        self.theta = [0.0, 0.0, 0.0]

        # Joint angle limits (min, max) for each joint
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi + 0.261, np.pi - 0.261],
            [0, self.l1 + self.l3 - self.l5],
        ]

        # End-effector (EE) object to store EE position and orientation
        self.ee = EndEffector()

        # Number of degrees of freedom and number of points to store robot configuration
        self.num_dof = 3
        self.num_points = 7
        self.points = [None] * self.num_points

        # Transformation matrices (DH parameters and resulting transformation)

        ########################################

        # insert your additional code here

        ########################################

    def calc_forward_kinematics(self, theta: list, radians=False):
        """
        Calculate Forward Kinematics (FK) based on the given joint angles.

        Args:
            theta (list): Joint angles (in radians if radians=True, otherwise in degrees).
            radians (bool): Whether the input angles are in radians (default is False).
        """
        ########################################

        ########################################

        # Calculate robot points (e.g., end-effector position)
        self.calc_robot_points()

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate Inverse Kinematics (IK) based on the input end-effector coordinates.

        Args:
            EE (EndEffector): End-effector object containing desired position (x, y, z).
            soln (int): Solution index (0 or 1), for multiple possible IK solutions.
        """
        x, y, z = EE.x, EE.y, EE.z
        l1, l2, l3, l4, l5 = self.l1, self.l2, self.l3, self.l4, self.l5

        ########################################

        # insert your code here

        ########################################

        # Recalculate Forward Kinematics to update the robot's configuration
        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate velocity kinematics and update joint velocities.

        Args:
            vel (array): Linear velocities (3D) of the end-effector.
        """
        ########################################

        # insert your code here

        ########################################

        # Recalculate robot points based on updated joint angles
        self.calc_robot_points()

    def calc_robot_points(self):
        """
        Calculate the main robot points (links and end-effector position) using the current joint angles.
        Updates the robot's points array and end-effector position.
        """

        # Calculate transformation matrices for each joint and end-effector
        self.points[0] = np.array([0, 0, 0, 1])
        self.points[1] = np.array([0, 0, self.l1, 1])
        self.points[2] = self.T[0] @ self.points[0]
        self.points[3] = self.points[2] + np.array([0, 0, self.l3, 1])
        self.points[4] = self.T[0] @ self.T[1] @ self.points[0] + np.array(
            [0, 0, self.l5, 1]
        )
        self.points[5] = self.T[0] @ self.T[1] @ self.points[0]
        self.points[6] = self.T[0] @ self.T[1] @ self.T[2] @ self.points[0]

        self.EE_axes = (
            self.T[0] @ self.T[1] @ self.T[2] @ np.array([0.075, 0.075, 0.075, 1])
        )
        self.T_ee = self.T[0] @ self.T[1] @ self.T[2]

        # End-effector (EE) position and axes
        self.ee.x = self.points[-1][0]
        self.ee.y = self.points[-1][1]
        self.ee.z = self.points[-1][2]
        rpy = rotm_to_euler(self.T_ee[:3, :3])
        self.ee.rotx, self.ee.roty, self.ee.rotz = rpy

        # EE coordinate axes
        self.EE_axes = np.zeros((3, 3))
        self.EE_axes[0] = self.T_ee[:3, 0] * 0.075 + self.points[-1][0:3]
        self.EE_axes[1] = self.T_ee[:3, 1] * 0.075 + self.points[-1][0:3]
        self.EE_axes[2] = self.T_ee[:3, 2] * 0.075 + self.points[-1][0:3]


class FiveDOFRobot:
    """
    A class to represent a 5-DOF robotic arm with kinematics calculations, including
    forward kinematics, inverse kinematics, velocity kinematics, and Jacobian computation.

    Attributes:
        l1, l2, l3, l4, l5: Link lengths of the robotic arm.
        theta: List of joint angles in radians.
        theta_limits: Joint limits for each joint.
        ee: End-effector object for storing the position and orientation of the end-effector.
        num_dof: Number of degrees of freedom (5 in this case).
        points: List storing the positions of the robot joints.
        DH: Denavit-Hartenberg parameters for each joint.
        T: Transformation matrices for each joint.
    """

    def __init__(self):
        """Initialize the robot parameters and joint limits."""
        # Link lengths
        self.l1, self.l2, self.l3, self.l4, self.l5 = 0.30, 0.15, 0.18, 0.15, 0.12

        # Joint angles (initialized to zero)
        self.theta = [0, 0, 0, 0, 0]

        # Joint limits (in radians)
        self.theta_limits = [
            [-np.pi, np.pi],
            [-np.pi / 3, np.pi],
            [-np.pi + np.pi / 12, np.pi - np.pi / 4],
            [-np.pi + np.pi / 12, np.pi - np.pi / 12],
            [-np.pi, np.pi],
        ]

        # End-effector object
        self.ee = EndEffector()

        # Robot's points
        self.num_dof = 5
        self.points = [None] * (self.num_dof + 1)

        # Denavit-Hartenberg parameters and transformation matrices

        self.H05 = np.matrix(
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        )  # Denavit-Hartenberg parameters (theta, d, a, alpha)
        # Transformation matrices

        # Initializes H matrices as empty
        self.H_01 = np.empty((4, 4))
        self.H_12 = np.empty((4, 4))
        self.H_23 = np.empty((4, 4))
        self.H_34 = np.empty((4, 4))
        self.H_45 = np.empty((4, 4))

        # initializes T matrix as empty
        self.T = np.zeros((self.num_dof, 4, 4))

    def calc_forward_kinematics(self, theta: list, radians=False, update_robot=True):
        """
        Calculate forward kinematics based on the provided joint angles.

        Args:
            theta: List of joint angles (in degrees or radians).
            radians: Boolean flag to indicate if input angles are in radians.
            update_robot: boolean flag to indicate whether the robot gets
                updated when calling this function. Used primarily to validate
                inverse kinematics solutions.
        """
        if not radians:
            # changes all angles to radians for math if the input was in degrees
            theta = [np.deg2rad(angle) for angle in theta]

        # calculates the dh table based on the passed in thetas
        self.H_01 = dh_to_matrix([theta[0], self.l1, 0, -PI / 2])
        self.H_12 = dh_to_matrix([theta[1] - PI / 2, 0, self.l2, PI])
        self.H_23 = dh_to_matrix([theta[2], 0, self.l3, PI])
        self.H_34 = dh_to_matrix([theta[3] + PI / 2, 0, 0, PI / 2])
        self.H_45 = dh_to_matrix([theta[4], self.l4 + self.l5, 0, 0])

        # calculates the full H and T matrix from base to EE
        self.H05 = self.H_01 @ self.H_12 @ self.H_23 @ self.H_34 @ self.H_45
        self.T = [self.H_01, self.H_12, self.H_23, self.H_34, self.H_45]

        # updates the actual robot thetas if desired
        if update_robot:
            self.theta = theta

        # Debug prints to verify the transformation matrices
        # print("Transformation Matrices:")
        # for i, T in enumerate(self.T):
        # print(f"T[{i}] =\n{T}")

        # Calculate robot points (positions of joints)
        self.calc_robot_points()

    def jacobian_v(self):
        """
        Given the current self.theta, uses the H matrices to calculate the
        jacobian matrix for the robot.

        Returns:
            J: np.array of the full jacobian matrix
        """

        # initializes the empty Jacobian matrix
        J = np.empty((3, 5))

        # calculates first column of matrix
        t1 = self.H05[0:3, 3]
        r1 = np.array([0, 0, 1])
        j1 = np.cross(r1, t1)
        J[0:3, 0] = j1

        # calculates second column of matrix
        t2 = self.H05[0:3, 3] - self.H_01[0:3, 3]
        r2 = self.H_01[0:3, 2]
        j2 = np.cross(r2, t2)
        J[0:3, 1] = j2

        # calculates third column of matrix
        t3 = self.H05[0:3, 3] - (self.H_01 @ self.H_12)[0:3, 3]
        r3 = (self.H_01 @ self.H_12)[0:3, 2]
        j3 = np.cross(r3, t3)
        J[0:3, 2] = j3

        # calculates fourth column of matrix
        t4 = self.H05[0:3, 3] - (self.H_01 @ self.H_12 @ self.H_23)[0:3, 3]
        r4 = (self.H_01 @ self.H_12 @ self.H_23)[0:3, 2]
        j4 = np.cross(r4, t4)
        J[0:3, 3] = j4

        # calculates fifth column of matrix
        t5 = self.H05[0:3, 3] - (self.H_01 @ self.H_12 @ self.H_23 @ self.H_34)[0:3, 3]
        r5 = (self.H_01 @ self.H_12 @ self.H_23 @ self.H_34)[0:3, 2]
        j5 = np.cross(r5, t5)
        J[0:3, 4] = j5

        return J

    def inverse_jacobian(self):
        """
        Creates the inverse jacobian matrix based on the jacobian.

        Returns:
            the pseudo inverse of the jacobian matrix
        """
        J = self.jacobian_v()

        # Calculate pseudo inverse of the jacobian matrix
        # The method used is to minimize the effects of singularities on
        # the robot
        lambda_constant = 0.01
        J_inv = np.transpose(J) @ np.linalg.inv(
            ((J @ np.transpose(J)) + lambda_constant**2 * np.identity(3))
        )

        return J_inv

    def calc_inverse_kinematics(self, EE: EndEffector, soln=0):
        """
        Calculate inverse kinematics to determine the joint angles based on end-effector position.

        Args:
            EE: EndEffector object containing desired position and orientation.
            soln: Optional parameter for multiple solutions (not implemented).
        """
        start_time = time.time()
        ########################################

        # print(f"{EE.x=}, {EE.y=}, {EE.z=}")

        # initializes our possible solution list
        possible_solutions = []
        for i in range(8):
            possible_solutions.append([0, 0, 0, 0, 0])

        # extract euler angles from our end effector object
        euler_angles = (EE.rotx, EE.roty, EE.rotz)
        # print(f"{euler_angles=}")

        # calculates rotation matrix from the the euler angles
        R = np.array(euler_to_rotm(euler_angles), dtype=float)
        # print(f"{R=}")

        # extract the end effector location from the ee object
        ee_location = np.array([EE.x, EE.y, EE.z], dtype=float)
        # print(f"{ee_location=}")

        # calculates the position of the wrist
        Pw = np.array([], dtype=float)
        Pw = ee_location - (self.l4 + self.l5) * (
            R @ np.transpose(np.array([0, 0, 1], dtype=float))
        )
        # print(f"{Pw=}")

        # calculates intermediary values used in calculating
        # thetas 1-3
        s = Pw[2] - self.l1
        r = np.sqrt((Pw[0] ** 2) + (Pw[1] ** 2))
        L = np.sqrt((s**2) + (r**2))
        beta = np.arccos(
            np.clip((self.l2**2 + self.l3**2 - L**2) / (2 * self.l2 * self.l3), -1, 1)
        )

        # Calculates all the possible solutions using 4 for loops
        # after all 4 loops, the goal is to build out all 8 possible solutions
        # The possible_solutions list contains all these values, in a structure
        # that looks like:
        # 1: [norm,   +, +, t4, t5]
        # 2: [norm,   +, -, t4, t5]
        # 3: [norm,   -, +, t4, t5]
        # 4: [norm,   -, -, t4, t5]
        # 5: [offset, +, +, t4, t5]
        # 6: [offset, +, -, t4, t5]
        # 7: [offset, -, +, t4, t5]
        # 8: [offset, -, -, t4, t5]
        # We have 8 possible permutations of thetas 1-3, then we calc theta 4,5
        # based on all those permutations

        # Calculates both possibilities of theta 1, and starts building
        # the possible_solutions list
        for i, solution in enumerate(possible_solutions):
            poss_soln = solution
            # cuts the 8 possible solutions in half, the first 4 getting one
            # possible theta 1 solution, and the last for getting the others
            if i < (len(possible_solutions) / 2):
                poss_soln[0] = float(np.arctan2(Pw[1], Pw[0]))
                # print(f"{i}: norm")
            else:
                poss_soln[0] = float(wraptopi(np.pi + np.arctan2(Pw[1], Pw[0])))
                # print(f"{i}: offset")

            # appends each theta 1 solution to all of the possible soln
            possible_solutions[i] = poss_soln

        # Calculates theta 3 for all of the possible solutions
        for i, solution in enumerate(possible_solutions):
            poss_soln = solution
            # alternates which theta 2 soln is calculated
            if i % 2 is 0:
                poss_soln[2] = float(-np.pi + beta)
            else:
                poss_soln[2] = float(np.pi - beta)

            # appends each theta 2 to the possible solutions
            possible_solutions[i] = poss_soln

        # Calculates theta 2 in the ++,-- order
        for i, solution in enumerate(possible_solutions):
            if i % 2 == 0:
                for j in range(2):
                    poss_soln = possible_solutions[2 * i + j]
                    poss_soln[1] = float(
                        np.arctan2(-r, s)
                        - np.arctan2(
                            self.l3 * np.sin(-poss_soln[2]),
                            (self.l2 + self.l3 * np.cos(-poss_soln[2])),
                        )
                    )
                    # print(f"{i}: neg")
                    possible_solutions[2 * i + j] = poss_soln
            else:
                for j in range(2):
                    poss_soln = possible_solutions[2 * i + j]
                    poss_soln[1] = float(
                        np.arctan2(r, s)
                        - np.arctan2(
                            self.l3 * np.sin(-poss_soln[2]),
                            (self.l2 + self.l3 * np.cos(-poss_soln[2])),
                        )
                    )
                    # print(f"{i}: pos")
                    possible_solutions[2 * i + j] = poss_soln
            if i == 3:
                break
            # print(f"{possible_solutions=}")

        # calculates thetas 4 and 5 based on thetas 1-3 for each possible soln
        for i, solution in enumerate(possible_solutions):

            t1 = solution[0]
            t2 = solution[1]
            t3 = solution[2]

            # creating rotation matrices for each part of the robot
            r_1 = np.array(
                [[np.cos(t1), 0, -np.sin(t1)], [np.sin(t1), 0, np.cos(t1)], [0, -1, 0]]
            )
            r_2 = np.array(
                [
                    [np.cos(t2 - np.pi / 2), np.sin(t2 - np.pi / 2), 0],
                    [np.sin(t2 - np.pi / 2), -np.cos(t2 - np.pi / 2), 0],
                    [0, 0, 1],
                ]
            )
            r_3 = np.array(
                [[np.cos(t3), np.sin(t3), 0], [np.sin(t3), -np.cos(t3), 0], [0, 0, 1]]
            )

            r_0_3 = r_1 @ r_2 @ r_3
            # print(f"{r_0_3=}")

            r_3_5 = np.transpose(r_0_3) @ R
            # print(f"{r_3_5=}")

            # calculates theta 4 and 5
            t4 = np.arctan2(r_3_5[1, 2], r_3_5[0, 2])
            # print(f"{t4=}")
            t5 = np.arctan2(r_3_5[2, 1], r_3_5[2, 0])
            # print(f"{t5=}")

            solution[3] = float(t4)
            solution[4] = float(t5)
            possible_solutions[i] = solution

        # uncomment below to print the thetas for each solution
        # for solution in possible_solutions:
        #     print(
        #         f"[{round(degrees(solution[0]), 4)}, {round(degrees(solution[1]), 4)}, {round(degrees(solution[2]), 4)}, {round(degrees(solution[3]), 4)}, {round(degrees(solution[4]), 4)}]"
        #     )

        # print(f"{possible_solutions=}")

        # goes through each possible solution calculated, then checks each one
        # for feasability and accuracy

        valid_solutions = []  # initializes the valid solutions

        for solution in possible_solutions:
            # calculates the forward kinematics, specifically without updating
            # the robot
            self.calc_forward_kinematics(solution, True, False)
            EE_points_desired = [EE.x, EE.y, EE.z]  # ee location inputted
            EE_points_calc = self.points[5][:3]  # ee location calculated
            # print(f"desired: {EE_points_desired},   calc: {EE_points_calc}")
            valid_thetas = [False] * 5
            for i, theta in enumerate(solution):
                # checks the possible solutions on 2 criteria:
                #   1: all the thetas are within the theta limits of the robot
                #   2: the solution is accurate based on the fpk solution
                if (
                    min(self.theta_limits[i][0], self.theta_limits[i][1])
                    < theta
                    < max(self.theta_limits[i][0], self.theta_limits[i][1])
                ) and (
                    all(
                        abs(EE_points_desired[k] - EE_points_calc[k]) <= 0.01
                        for k in range(3)
                    )
                ):
                    valid_thetas[i] = True
            # after checking the criteria for all thetas, append to the valid
            # solutions list
            if all(valid_thetas):
                valid_solutions.append(solution)

        # print(f"{valid_solutions=}")

        # This block tells the visulaizer which solutions to show,
        # if only 1 valid solutions exists, it always shows that one
        # if there are 2 or more, it shows the first 2 valid solns, but
        # if you know that there are more valid soln, change the indexing to
        # visulaize the other solutions.
        if soln == 0 or len(valid_solutions) == 1:
            self.theta = valid_solutions[0]
        else:
            self.theta = valid_solutions[1]

        # updates the robot with the valid solutions.
        self.calc_forward_kinematics(self.theta, radians=True)

        EE_points_desired = [EE.x, EE.y, EE.z]  # ee location inputted
        EE_points_calc = self.points[5][:3]  # ee location calculated
        err = abs(EE_points_desired - EE_points_calc)
        print(f"Anayltical IK error: {err}")
        end_time = time.time()
        ex_time = end_time - start_time
        print(f"Analytical IK took {ex_time:.5f} second to run")

        ########################################

    def calc_numerical_ik(self, EE: EndEffector, tol=0.005, ilimit=100):
        """
        Calculate numerical inverse kinematics based on input coordinates.

        Args:
            EE: EndEffector object containing the desired ee location
            tol: float representing how close the code needs to get to the
                actual ee location before it can complete
            ilimit: int representing the max number of iterations before it
                gives up on finding a better solution
        """
        start_time = time.time()

        x_ee, y_ee, z_ee = EE.x, EE.y, EE.z

        ########################################
        # sets current thetas to theta, which is used in calculations
        theta = self.theta

        i = 0

        # iterates through until we reach the max iteration number
        while i < ilimit:
            self.calc_forward_kinematics(theta, radians=True)
            # calculates the error between
            err = np.transpose([x_ee - self.ee.x, y_ee - self.ee.y, z_ee - self.ee.z])

            # checks to see if the ee is close enough to the desired location
            if np.linalg.norm(err) > tol:
                # if the error is too big, calculate the next step closer to
                # the desired ee location
                theta = theta + (self.inverse_jacobian() @ err) * 0.1
                # print(self.inverse_jacobian() @ err)
            else:
                # if the error is within the tolerance, then break the loop
                # print("position found")
                self.theta = theta
                break

            i += 1

        ########################################

        self.calc_robot_points()

        ########################################
        self.calc_forward_kinematics(self.theta, radians=True)

        EE_points_desired = [EE.x, EE.y, EE.z]  # ee location inputted
        EE_points_calc = self.points[5][:3]  # ee location calculated
        err = abs(EE_points_desired - EE_points_calc)
        print(f"Numerical IK error: {err}")

        end_time = time.time()
        ex_time = end_time - start_time
        print(f"Numerical IK took {ex_time:.5f} second to run")

    def calc_velocity_kinematics(self, vel: list):
        """
        Calculate the joint velocities required to achieve the given end-effector velocity.

        Args:
            vel: Desired end-effector velocity (3x1 vector).
        """
        ########################################
        # at every time step, inverse inverse jacboan * cartesian
        time_step = 0.01

        q_dot = self.inverse_jacobian() @ np.array(vel)
        self.theta = self.theta + time_step * np.array(q_dot)

        ########################################

        # Recompute robot points based on updated joint angles
        self.calc_forward_kinematics(self.theta, radians=True)

    def calc_robot_points(self, update_robot=True):
        """Calculates the main arm points using the current joint angles"""

        # Initialize points[0] to the base (origin)
        self.points[0] = np.array([0, 0, 0, 1])

        # Precompute cumulative transformations to avoid redundant calculations
        T_cumulative = [np.eye(4)]
        for i in range(self.num_dof):
            T_cumulative.append(T_cumulative[-1] @ self.T[i])

        # Calculate the robot points by applying the cumulative transformations
        for i in range(1, 6):
            self.points[i] = T_cumulative[i] @ self.points[0]

        if update_robot:
            # Calculate EE position and rotation
            self.EE_axes = T_cumulative[-1] @ np.array(
                [0.075, 0.075, 0.075, 1]
            )  # End-effector axes
            self.T_ee = T_cumulative[-1]  # Final transformation matrix for EE

            # Set the end effector (EE) position
            self.ee.x, self.ee.y, self.ee.z = self.points[-1][
                :3
            ]  # pylint: disable=unsubscriptable-object
            # Extract and assign the RPY (roll, pitch, yaw) from the rotation matrix
            rpy = rotm_to_euler(self.T_ee[:3, :3])
            self.ee.rotx, self.ee.roty, self.ee.rotz = rpy[0], rpy[1], rpy[2]

            # Calculate the EE axes in space (in the base frame)
            self.EE = [self.ee.x, self.ee.y, self.ee.z]
            self.EE_axes = np.array(
                [
                    self.T_ee[:3, i] * 0.075 + self.points[-1][:3] for i in range(3)
                ]  # pylint: disable=unsubscriptable-object
            )

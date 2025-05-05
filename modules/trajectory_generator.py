import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import yaml


import modules.arm_models as arm
from helper_fcns.utils import EndEffector


class MultiAxisTrajectoryGenerator:
    """
    Multi-axis trajectory generator for joint or task space trajectories.

    Supports linear, cubic, quintic polynomial, and trapezoidal velocity profiles.
    """

    def __init__(
        self,
        method="quintic",
        mode="joint",
        interval=[0, 1],
        ndof=1,
        start_pos=None,
        final_pos=None,
        start_vel=None,
        final_vel=None,
        start_acc=None,
        final_acc=None,
    ):
        """
        Initialize the trajectory generator with the given configuration.

        Args:
            method (str): Type of trajectory ('linear', 'cubic', 'quintic', 'trapezoid').
            mode (str): 'joint' for joint space, 'task' for task space.
            interval (list): Time interval [start, end] in seconds.
            ndof (int): Number of degrees of freedom.
            start_pos (list): Initial positions.
            final_pos (list): Final positions.
            start_vel (list): Initial velocities (default 0).
            final_vel (list): Final velocities (default 0).
            start_acc (list): Initial accelerations (default 0).
            final_acc (list): Final accelerations (default 0).
        """

        self.T = interval[1]
        self.ndof = ndof
        self.t = None

        if mode == "joint":
            self.mode = "Joint Space"
            # self.labels = ['th1', 'th2', 'th3', 'th4', 'th5']
            self.labels = [f"axis{i+1}" for i in range(self.ndof)]
            self.labels = [f"axis{i+1}" for i in range(self.ndof)]
        elif mode == "task":
            self.mode = "Task Space"
            self.labels = ["x", "y", "z"]

            self.labels = ["x", "y", "z"]

        # Assign positions and boundary conditions
        self.start_pos = start_pos
        self.final_pos = final_pos
        self.start_vel = start_vel if start_vel is not None else [0] * self.ndof
        self.final_vel = final_vel if final_vel is not None else [0] * self.ndof
        self.start_acc = start_acc if start_acc is not None else [0] * self.ndof
        self.final_acc = final_acc if final_acc is not None else [0] * self.ndof
        self.final_acc = final_acc if final_acc is not None else [0] * self.ndof

        # Select trajectory generation method
        if method == "linear":
            self.m = LinearInterp(self)
        elif method == "cubic":
            self.m = CubicPolynomial(self)
        elif method == "quintic":
            self.m = QuinticPolynomial(self)
        elif method == "septic":
            self.m = SepticPolynomial(self)
        elif method == "trapezoid":
            self.m = TrapezoidVelocity(self)
        elif method == "spline":
            self.m = Spline(self)

    def generate(self, nsteps=100):
        """
        Generate the trajectory at discrete time steps.

        Args:
            nsteps (int): Number of time steps.
        Returns:
            list: List of position, velocity, acceleration for each DOF.
        """
        self.t = np.linspace(0, self.T, nsteps)
        return self.m.generate(nsteps=nsteps)

    def plot(self):
        """
        Plot the position, velocity, and acceleration trajectories.
        """
        self.fig = plt.figure()
        self.sub1 = self.fig.add_subplot(3, 1, 1)  # Position plot
        self.sub2 = self.fig.add_subplot(3, 1, 2)  # Velocity plot
        self.sub3 = self.fig.add_subplot(3, 1, 3)  # Acceleration plot

        self.fig.set_size_inches(8, 10)
        self.fig.suptitle(self.mode + " Trajectory Generator", fontsize=16)

        colors = ["r", "g", "b", "m", "y"]

        for i in range(self.ndof):
            # position plot
            self.sub1.plot(
                self.t, self.m.X[i][0], colors[i] + "o-", label=self.labels[i]
            )
            self.sub1.set_ylabel("position", fontsize=15)
            self.sub1.grid(True)
            self.sub1.legend()

            # velocity plot
            self.sub2.plot(
                self.t, self.m.X[i][1], colors[i] + "o-", label=self.labels[i]
            )
            self.sub2.set_ylabel("velocity", fontsize=15)
            self.sub2.grid(True)
            self.sub2.legend()

            # acceleration plot
            self.sub3.plot(
                self.t, self.m.X[i][2], colors[i] + "o-", label=self.labels[i]
            )
            self.sub3.set_ylabel("acceleration", fontsize=15)
            self.sub3.set_xlabel("Time (secs)", fontsize=18)
            self.sub3.grid(True)
            self.sub3.legend()

        plt.show()


class LinearInterp:
    """
    Linear interpolation between start and end positions.
    """

    def __init__(self, trajgen):
        self._copy_params(trajgen)
        self.solve()

    def _copy_params(self, trajgen):
        self.start_pos = trajgen.start_pos
        self.final_pos = trajgen.final_pos
        self.T = trajgen.T
        self.ndof = trajgen.ndof
        self.X = [None] * self.ndof

    def solve(self):
        pass  # Linear interpolation is directly computed in generate()

    def generate(self, nsteps=100):

        self.t = np.linspace(0, self.T, nsteps)
        for i in range(self.ndof):  # iterate through all DOFs
            q, qd, qdd = [], [], []
            for t in self.t:  # iterate through time, t
                q.append(
                    (1 - t / self.T) * self.start_pos[i]
                    + (t / self.T) * self.final_pos[i]
                )
                qd.append(self.final_pos[i] - self.start_pos[i])
                qdd.append(0)
                qdd.append(0)
            self.X[i] = [q, qd, qdd]
        return self.X


class CubicPolynomial:
    """
    Cubic interpolation with position and velocity boundary constraints.
    """

    def __init__(self, trajgen):
        self._copy_params(trajgen)
        self.solve()

    def _copy_params(self, trajgen):
        self.start_pos = trajgen.start_pos
        self.start_vel = trajgen.start_vel
        self.final_pos = trajgen.final_pos
        self.final_vel = trajgen.final_vel
        self.T = trajgen.T
        self.ndof = trajgen.ndof
        self.X = [None] * self.ndof

    def solve(self):
        t0, tf = 0, self.T
        self.A = np.array(
            [
                [1, t0, t0**2, t0**3],
                [0, 1, 2 * t0, 3 * t0**2],
                [1, tf, tf**2, tf**3],
                [0, 1, 2 * tf, 3 * tf**2],
            ]
        )
        self.b = np.zeros([4, self.ndof])

        for i in range(self.ndof):
            self.b[:, i] = [
                self.start_pos[i],
                self.start_vel[i],
                self.final_pos[i],
                self.final_vel[i],
            ]

        self.coeff = np.linalg.solve(self.A, self.b)

    def generate(self, nsteps=100):
        self.t = np.linspace(0, self.T, nsteps)

        for i in range(self.ndof):  # iterate through all DOFs
            q, qd, qdd = [], [], []
            c = self.coeff[:, i]
            for t in self.t:  # iterate through time, t
                q.append(c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3)
                qd.append(c[1] + 2 * c[2] * t + 3 * c[3] * t**2)
                qdd.append(2 * c[2] + 6 * c[3] * t)
                qdd.append(2 * c[2] + 6 * c[3] * t)
            self.X[i] = [q, qd, qdd]
        return self.X


class QuinticPolynomial:
    """
    Quintic interpolation with position, velocity, and acceleration constraints.
    """

    def __init__(self, trajgen):
        self._copy_params(trajgen)
        self.solve()

    def _copy_params(self, trajgen):
        self.start_pos = trajgen.start_pos
        self.start_vel = trajgen.start_vel
        self.start_acc = trajgen.start_acc
        self.final_pos = trajgen.final_pos
        self.final_vel = trajgen.final_vel
        self.final_acc = trajgen.final_acc
        self.T = trajgen.T
        self.ndof = trajgen.ndof
        self.X = [None] * self.ndof

    def solve(self):
        t0, tf = 0, self.T
        self.A = np.array(
            [
                [1, t0, t0**2, t0**3, t0**4, t0 * 5],
                [0, 1, 2 * t0, 3 * t0**2, 4 * t0**3, 5 * t0**4],
                [0, 0, 2, 6 * t0, 12 * t0**2, 20 * t0**3],
                [1, tf, tf**2, tf**3, tf**4, tf**5],
                [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
            ]
        )
        self.b = np.zeros([6, self.ndof])

        for i in range(self.ndof):
            self.b[:, i] = [
                self.start_pos[i],
                self.start_vel[i],
                self.start_acc[i],
                self.final_pos[i],
                self.final_vel[i],
                self.final_acc[i],
            ]

        self.coeff = np.linalg.solve(self.A, self.b)

    def generate(self, nsteps=100):
        self.t = np.linspace(0, self.T, nsteps)

        for i in range(self.ndof):  # iterate through all DOFs
            q, qd, qdd = [], [], []
            c = self.coeff[:, i]
            for t in self.t:  # iterate through time, t
                q.append(
                    c[0]
                    + c[1] * t
                    + c[2] * t**2
                    + c[3] * t**3
                    + c[4] * t**4
                    + c[5] * t**5
                )
                qd.append(
                    c[1]
                    + 2 * c[2] * t
                    + 3 * c[3] * t**2
                    + 4 * c[4] * t**3
                    + 5 * c[5] * t**4
                )
                qdd.append(
                    2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3
                )
            self.X[i] = [q, qd, qdd]
        return self.X


class SepticPolynomial:
    """
    Septic interpolation with position, velocity, and acceleration constraints.
    """

    def __init__(self, trajgen):
        self._copy_params(trajgen)
        self.solve()

    def _copy_params(self, trajgen):
        self.start_pos = trajgen.start_pos
        self.start_vel = trajgen.start_vel
        self.start_acc = trajgen.start_acc
        self.final_pos = trajgen.final_pos
        self.final_vel = trajgen.final_vel
        self.final_acc = trajgen.final_acc
        self.T = trajgen.T
        self.ndof = trajgen.ndof
        self.X = [None] * self.ndof

    def solve(self):

        # extract the initial and final time
        t0, tf = 0, self.T
        # print(f"start and end: \n{self.start_pos}\n{self.final_pos}")
        # populate the A matrix
        self.A = np.array(
            [
                [1, t0, t0**2, t0**3, t0**4, t0**5, t0**6, t0**7],  # q(t0)
                [
                    0,
                    1,
                    2 * t0,
                    3 * t0**2,
                    4 * t0**3,
                    5 * t0**4,
                    6 * t0**5,
                    7 * t0**6,
                ],  # v(t0)
                [
                    0,
                    0,
                    2,
                    6 * t0,
                    12 * t0**2,
                    20 * t0**3,
                    30 * t0**4,
                    42 * t0**5,
                ],  # a(t0)
                [0, 0, 0, 6, 24 * t0, 60 * t0**2, 120 * t0**3, 210 * t0**4],  # j(t0)
                [1, tf, tf**2, tf**3, tf**4, tf**5, tf**6, tf**7],  # q(tf)
                [
                    0,
                    1,
                    2 * tf,
                    3 * tf**2,
                    4 * tf**3,
                    5 * tf**4,
                    6 * tf**5,
                    7 * tf**6,
                ],  # v(tf)
                [
                    0,
                    0,
                    2,
                    6 * tf,
                    12 * tf**2,
                    20 * tf**3,
                    30 * tf**4,
                    42 * tf**5,
                ],  # a(tf)
                [0, 0, 0, 6, 24 * tf, 60 * tf**2, 120 * tf**3, 210 * tf**4],  # j(tf)
            ]
        )

        # populate the b matrix ()
        self.b = np.zeros([8, self.ndof])
        for i in range(self.ndof):
            self.b[:, i] = [
                self.start_pos[i],
                self.start_vel[i],
                self.start_acc[i],
                0,  # start jerk 0
                self.final_pos[i],
                self.final_vel[i],
                self.final_acc[i],
                0,  # set final jerk to 0
            ]

        # print("self.b", self.b)
        # solve for the coefficients
        self.coeff = np.linalg.solve(self.A, self.b)
        # print("size self.coeff", self.coeff)

    def generate(self, nsteps=100):
        self.t = np.linspace(0, self.T, nsteps)

        for i in range(self.ndof):  # iterate through all DOFs
            q, qd, qdd, qddd = [], [], [], []
            c = self.coeff[:, i]
            for t in self.t:  # iterate through time, t
                q.append(
                    c[0]
                    + c[1] * t
                    + c[2] * t**2
                    + c[3] * t**3
                    + c[4] * t**4
                    + c[5] * t**5
                    + c[6] * t**6
                    + c[7] * t**7
                )
                qd.append(
                    c[1]
                    + 2 * c[2] * t
                    + 3 * c[3] * t**2
                    + 4 * c[4] * t**3
                    + 5 * c[5] * t**4
                    + 6 * c[6] * t**5
                    + 7 * c[7] * t**6
                )
                qdd.append(
                    2 * c[2]
                    + 6 * c[3] * t
                    + 12 * c[4] * t**2
                    + 20 * c[5] * t**3
                    + 30 * c[6] * t**4
                    + 42 * c[7] * t**5
                )
                qddd.append(
                    6 * c[3]
                    + 24 * c[4] * t
                    + 60 * c[5] * t**2
                    + 120 * c[6] * t**3
                    + 210 * c[7] * t**4
                )
            self.X[i] = [q, qd, qdd]  # , qddd]
        # y = self.X
        return self.X


class TrapezoidVelocity:
    """
    Trapezoidal velocity profile generator for constant acceleration/deceleration phases.
    """

    def __init__(self, trajgen):
        self._copy_params(trajgen)
        self.solve()

    def _copy_params(self, trajgen):
        self.start_pos = trajgen.start_pos
        self.start_vel = trajgen.start_vel
        self.final_pos = trajgen.final_pos
        self.final_vel = trajgen.final_vel
        self.T = trajgen.T
        self.ndof = trajgen.ndof
        self.X = [None] * self.ndof

    def solve(self):

        # note system constraints
        max_vel = 10
        max_acc = 6

        # extract travel time
        t0, tf = 0, self.T
        delta_t = tf - t0

        peak_vel_vec = []
        peak_acc_vec = []
        t_blend_vec = []

        for j in range(self.ndof):

            # extract stast and final coord in dimension j
            q0, qf = self.start_pos[j], self.final_pos[j]
            delta_q = qf - q0

            if delta_q == 0:
                peak_vel_vec.append(0)
                peak_acc_vec.append(0)
                t_blend_vec.append(0)
            else:
                # calc peak velocity and time to blend
                peak_vel = 2 * delta_q / tf
                t_blend = (-delta_q + peak_vel * tf) / peak_vel

                # check if t_blend is too long
                if (2 * t_blend) > delta_t:
                    # if t_blend is too long, scale vel
                    peak_vel = 0.5 * delta_t * max_acc

                    # re-calc t_blend
                    t_blend = (-delta_q + peak_vel * tf) / peak_vel

                # ignore postential issues w/ this for now
                peak_acc = peak_vel / t_blend

                # store answer
                peak_vel_vec.append(peak_vel)
                peak_acc_vec.append(peak_acc)
                t_blend_vec.append(t_blend)

        # return the calculated parameters
        self.trap_traj_params = [peak_vel_vec, peak_acc_vec, t_blend_vec]

    def generate(self, nsteps=100):

        # extract parameters
        vel_vec, acc_vec, tb_vec = self.trap_traj_params
        t0, tf = 0, self.T

        # get time vector
        self.t = np.linspace(t0, tf, nsteps)

        # calculate and store position, velocity, and acceleration
        # at each time in time_vec
        # separate into the three phases: acceleration, constant, deceleration

        for i in range(self.ndof):  # iterate through all DOFs
            # make containers
            q, qd, qdd = [], [], []
            q0, qf = self.start_pos[i], self.final_pos[i]

            # get vel, acc, and tb for the dof i
            vel = vel_vec[i]
            acc = acc_vec[i]
            tb = tb_vec[i]

            for t in self.t:
                # acceleration
                if t <= tb:
                    q.append(q0 + 0.5 * acc * (t**2))
                    qd.append(acc * t)
                    qdd.append(acc)

                # constant
                elif (t > tb) and (t <= (tf - tb)):
                    eqn1 = 0.5 * (qf + q0 - vel * tf) + (vel * t)
                    q.append(eqn1)
                    qd.append(vel)
                    qdd.append(acc)

                # deceleration
                else:
                    eqn2 = qf - 0.5 * acc * (tf**2) + acc * tf * t - 0.5 * acc * (t**2)
                    q.append(eqn2)
                    qd.append(acc * tf - acc * t)
                    qdd.append(-acc)

            self.X[i] = [q, qd, qdd]

        # return the pos, vel, and acc info for every dimension
        # and timestep here
        return self.X


class Spline:
    """
    Spline interpolation for smooth trajectories.
    """

    def __init__(self, trajgen):
        self._copy_params(trajgen)

    def _copy_params(self, trajgen):
        self.start_pos = trajgen.start_pos
        self.final_pos = trajgen.final_pos
        self.T = trajgen.T
        self.ndof = trajgen.ndof
        self.X = [None] * self.ndof

    def generate(self, nsteps=100):

        # create a 3d spline from the start point to the end point
        # using cubic spline interpolation

        with open("waypoints.yml", "r") as file:
            waypoints_pre = yaml.safe_load(file)
        waypoints = waypoints_pre["points"]

        way_x = [p[0] for p in waypoints]
        way_y = [p[1] for p in waypoints]
        way_z = [p[2] for p in waypoints]

        t_waypoints = np.linspace(0, self.T, len(waypoints))
        t = np.linspace(0, self.T, nsteps)
        spline_x = CubicSpline(t_waypoints, way_x)
        spline_y = CubicSpline(t_waypoints, way_y)
        spline_z = CubicSpline(t_waypoints, way_z)

        points = []
        for i in range(nsteps):
            points.append(
                [
                    round(float(spline_x(t)[i]), 4),
                    round(float(spline_y(t)[i]), 4),
                    round(float(spline_z(t)[i]), 4),
                ]
            )
            # print(f"\n\n{points[i]}\n\n")

        # combines the positions, velocities, and accels into q, qd, qdd,
        # and returns them
        # q, qd, qdd = [], [], []
        # for i in range(self.ndof):
        #     for point in points:
        #         q.append(point)
        #         qd.append([0, 0, 0])
        #         qdd.append([0, 0, 0])
        #     self.X.append([q, qd, qdd])

        # print(f"\n\n{self.X=}\n\n")

        # print(f"\n\nPoints: {points}\n\n")
        for i in range(self.ndof):  # iterate through all DOFs
            q, qd, qdd = [], [], []
            for j in range(nsteps):  # iterate through time, t
                # print(f"{i=} {j=}")
                q.append(points[j][i])
            qd = np.gradient(q, t)
            qdd = np.gradient(qd, t)
            self.X[i] = [q, qd, qdd]
        return self.X

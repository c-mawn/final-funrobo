import time
import matplotlib.pyplot as plt
from modules.trajectory_generator import *


def main():
    """Main function that runs the simulation"""
    modes = ["task", "joint"]
    methods = ["cubic", "quintic", "septic", "trapezoid"]
    elapsed = []
    labels = []
    for method in methods:
        for mode in modes:
            labels.append(":".join([method, mode]))
            times = []
            if mode == "task":
                q0 = [0.15, 0.15, 0.35]
                qf = [0.05, -0.25, 0.4]
            else:
                q0 = [44.99, -0.9, 38.32, -41.09, 0.44]
                qf = [-78.69, -24.54, 22.83, -10.93, 0.34]
            for _ in range(50):
                start_time = time.time()
                traj = MultiAxisTrajectoryGenerator(
                    method=method,
                    mode=mode,
                    interval=[0, 1],
                    ndof=len(q0),
                    start_pos=q0,
                    final_pos=qf,
                )

                # generate trajectory
                t = traj.generate(nsteps=50)

                time_elapsed = time.time() - start_time
                times.append(time_elapsed)
            elapsed.append(times)
    # print(elapsed)

    # plot elapsed times
    plot(elapsed, labels)

    # plot trajectory
    # traj.plot()


def plot(elapsed_times: list[list[float]], labels: list[str]):
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Create a bar plot
    ax.bar(labels, [sum(times) / len(times) for times in elapsed_times])
    # Set the title and labels
    ax.set_title("Trajectory Generation Time")
    ax.set_xlabel("Trajectory Generation Method")
    ax.set_ylabel("Time (s)")
    # Rotate the x-axis labels
    plt.xticks(rotation=45)
    # Show the plot
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

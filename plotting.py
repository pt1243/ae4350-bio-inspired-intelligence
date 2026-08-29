import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from evaluation import run_episode_with_trajectory
from environment import LunarHazardEnvironment


def plot_trajectory(episode_data, map_half_width, show_actual_position=True):
    """
    Plot the hazard map together with:
      - actual horizontal position over time
      - predicted touchdown point over time
      - nominal target
      - final touchdown point
    """

    positions = episode_data["positions"]
    predictions = episode_data["touchdown_predictions"]
    hazard_map = episode_data["hazard_map"]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Hazard map
    image = ax.imshow(
        hazard_map,
        extent=[-map_half_width, map_half_width, -map_half_width, map_half_width],
        origin="lower",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(image, ax=ax, label="Landing safety")

    # Predicted touchdown point trajectory
    ax.plot(predictions[:, 0], predictions[:, 1], label="Predicted touchdown", linewidth=2)

    # Physical horizontal position
    if show_actual_position:
        ax.plot(positions[:, 0], positions[:, 1], linestyle="--", label="Lander position")

    # Nominal landing target
    ax.scatter(0.0, 0.0, marker="x", s=100, label="Original target")

    # Initial predicted touchdown
    ax.scatter(predictions[0, 0], predictions[0, 1], marker="o", s=70, label="Initial prediction")

    # Actual final touchdown
    ax.scatter(positions[-1, 0], positions[-1, 1], marker="*", s=150, label="Touchdown")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"PPO terminal descent trajectory\nReturn = {episode_data['return']:.1f}")

    ax.set_xlim(-map_half_width, map_half_width)
    ax.set_ylim(-map_half_width, map_half_width)

    ax.set_aspect("equal")
    ax.legend()

    fig.tight_layout()

    return fig, ax


def plot_time_markers(ax, episode_data, interval=20.0):
    predictions = episode_data["touchdown_predictions"]
    times = episode_data["times"]

    next_time = interval

    for i, time in enumerate(times):
        if time >= next_time:
            ax.scatter(predictions[i, 0], predictions[i, 1], s=30)

            ax.annotate(
                f"{time:.0f}s",
                (
                    predictions[i, 0],
                    predictions[i, 1],
                ),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

            next_time += interval


def plot_control_effort(episode_data):
    times = episode_data["times"][:-1]

    actions = episode_data["actions"]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(times, actions[:, 0], label="Horizontal")
    ax.plot(times, actions[:, 1], label="Vertical")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration output [-]")
    ax.set_title("Control effort during descent")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    return fig, ax

def plot_learning_curve():
    data = np.load("learning_curve_ppo_lunar.npz")

    steps = data["steps"]
    means = data["mean_returns"]
    stds = data["std_returns"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, means, marker="o", label="Mean evaluation return")
    ax.fill_between(steps, means - stds, means + stds, alpha=0.2, label="±1 standard deviation")
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Mean episode returns")
    ax.set_title("PPO learning curve")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

plot_learning_curve()


env = LunarHazardEnvironment()
model = PPO.load("ppo_lunar")

for i in range(3):
    episode_data = run_episode_with_trajectory(
        env,
        model,
        seed=500+i,
    )
    plot_control_effort(episode_data)

    fig, ax = plot_trajectory(episode_data, env.map_half_width, show_actual_position=False)
    plot_time_markers(ax, episode_data, interval=20.0)

plt.show()

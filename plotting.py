from itertools import count

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from evaluation import run_episode_with_trajectory
from environment import LunarHazardEnvironment


def plot_trajectory(fig, ax, episode_data, map_half_width, show_actual_position=True):
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

    # Hazard map
    image = ax.imshow(
        hazard_map,
        extent=[-map_half_width, map_half_width, -map_half_width, map_half_width],
        origin="lower",
        vmin=0,
        vmax=1,
    )

    fig.colorbar(image, ax=ax, label="Safety score")

    # Nominal landing target
    ax.scatter(0.0, 0.0, marker="x", s=100, c="k", label="Nominal target")

    # Initial predicted touchdown
    ax.scatter(predictions[0, 0], predictions[0, 1], marker="x", s=70, c="tab:red", zorder=9, label="Initial predicted touchdown")
    # Predicted touchdown point trajectory
    ax.plot(predictions[:, 0], predictions[:, 1], color="tab:pink", label="Predicted touchdown position", linewidth=2)

    # Physical horizontal position
    if show_actual_position:
        ax.plot(positions[:, 0], positions[:, 1], linestyle="--", label="Lander position")



    # Actual final touchdown
    ax.scatter(positions[-1, 0], positions[-1, 1], marker="x", s=70, c="C0", zorder=10, label="Final touchdown position")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Trajectory")

    ax.set_xlim(-map_half_width, map_half_width)
    ax.set_ylim(-map_half_width, map_half_width)

    ax.set_aspect("equal")
    ax.legend()



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


def plot_control_effort(ax, episode_data):
    times = episode_data["times"][:-1]

    actions = episode_data["actions"]

    ax.plot(times, actions[:, 0], label="Horizontal")
    ax.plot(times, actions[:, 1], label="Vertical")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration output [-]")
    ax.set_title("Control effort during descent")
    ax.set_ylim(-1.1, 1.1)

    ax.grid(True)
    ax.legend(ncols=2)


def plot_training_progress():
    data = np.load("learning_curve_base_model.npz")

    steps = data["steps"]
    means_return = data["mean_returns"]
    stds_return = data["std_returns"]
    means_safety = data["mean_safeties"]
    stds_safety = data["std_safeties"]
    means_speed = data["mean_speeds"]
    stds_speed = data["std_speeds"]
    means_effort = data["mean_control_efforts"]
    stds_effort = data["std_control_efforts"]
    means_distance = data["mean_target_errors"]
    stds_distance = data["std_target_errors"]
    min_safety = data["min_safeties"]

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 7), sharex=True)

    # ---------------------------------------------------------
    # Episode return
    # ---------------------------------------------------------
    axes[0].plot(steps, means_return, label="Mean")

    axes[0].fill_between(
        steps, means_return - stds_return, means_return + stds_return, alpha=0.2, label="±1 standard deviation"
    )

    axes[0].set_ylabel("Total reward [-]")
    # axes[0].set_title("PPO training performance")
    axes[0].grid(True)

    # ---------------------------------------------------------
    # Touchdown safety
    # ---------------------------------------------------------
    axes[1].plot(steps, means_safety)

    axes[1].fill_between(steps, means_safety - stds_safety, means_safety + stds_safety, alpha=0.2)

    axes[1].set_ylabel("Safety score [-]")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    axes[1].plot(steps, min_safety, linestyle="--", c="r", label="Minimum safety score")

    # ---------------------------------------------------------
    # Touchdown velocity
    # ---------------------------------------------------------
    axes[2].plot(steps, means_speed)

    axes[2].fill_between(steps, means_speed - stds_speed, means_speed + stds_speed, alpha=0.2)

    axes[2].set_ylabel("Touchdown speed\n[m/s]")
    axes[2].grid(True)

    # ---------------------------------------------------------
    # Control effort
    # ---------------------------------------------------------
    axes[3].plot(steps, means_effort)

    axes[3].fill_between(steps, means_effort - stds_effort, means_effort + stds_effort, alpha=0.2)

    axes[3].set_ylabel("Control effort\n[m$^2$/s$^3$]")
    axes[3].grid(True)

    # ---------------------------------------------------------
    # Target distance
    # ---------------------------------------------------------
    axes[4].plot(steps, means_distance)

    axes[4].fill_between(steps, means_distance - stds_distance, means_distance + stds_distance, alpha=0.2)

    axes[4].set_ylabel("Target error [m]")
    axes[4].set_xlabel("Training steps [-]")
    axes[4].grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    handles_1, labels_1 = axes[1].get_legend_handles_labels()
    

    fig.legend(handles + handles_1, labels + labels_1, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01))

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig("training_progress.pdf", bbox_inches="tight", dpi=300)


plot_training_progress()


env = LunarHazardEnvironment()
model = PPO.load("base_model")

# for i in count():
#     episode_data = run_episode_with_trajectory(
#         env,
#         model,
#         seed=500 + i,
#     )
#     plot_control_effort(episode_data)

#     fig, ax = plot_trajectory(episode_data, env.map_half_width, show_actual_position=False, title=f" seed {500+i}")
    # plot_time_markers(ax, episode_data, interval=10.0)


# 506 - large movement
# 533 - squiggling
# 540 - nice long trajectory
# 543 - nice and smooth
# 546 - starts in a central low depression
for seed in [533, 543, 546]:
    episode_data = run_episode_with_trajectory(
        env, model, seed=seed
    )
    print(f"Seed {seed}: return = {episode_data["return"]:.2f}")
    fig, axs = plt.subplots(2, 1, figsize=(6, 7), height_ratios=[4, 1])
    plot_trajectory(fig, axs[0], episode_data, env.map_half_width, show_actual_position=False)
    plot_control_effort(axs[1], episode_data)
    fig.tight_layout()
    fig.savefig(f"trajectory_seed_{seed}.pdf", bbox_inches="tight", dpi=300)

# plt.show()

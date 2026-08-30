import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

from environment import LunarHazardEnvironment


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "base_model"

N_EVAL_EPISODES = 100
EVAL_SEED_START = 30_000

# Nominal training/environment conditions
NOMINAL_DOWNRANGE_VELOCITY = 0.25
NOMINAL_MAX_ACCELERATION = 0.10
NOMINAL_DESCENT_TIME = 60

# Robustness sweep ranges
DOWNRANGE_VELOCITIES = [0.10, 0.25, 0.40, 0.80]

MAX_ACCELERATIONS = [0.001, 0.01, 0.05, 0.10, 0.15, 0.30]

DESCENT_TIMES = [10, 20, 40, 60, 80]


# ============================================================
# Fixed-policy evaluation
# ============================================================


def evaluate_fixed_policy(
    model,
    env_params,
    n_episodes=N_EVAL_EPISODES,
):
    """
    Evaluate one fixed trained PPO policy under modified
    environment/dynamics parameters.
    """

    env = LunarHazardEnvironment(**env_params)

    safeties = []
    speeds = []
    target_errors = []
    control_efforts = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=EVAL_SEED_START + episode)

        terminated = False
        truncated = False
        control_effort = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            acceleration = action * env.max_acceleration

            control_effort += np.sum(acceleration**2) * env.dt

            (
                obs,
                reward,
                terminated,
                truncated,
                info,
            ) = env.step(action)

        safeties.append(info["touchdown_safety"])

        speeds.append(info["touchdown_speed"])

        target_errors.append(info["target_error"])

        control_efforts.append(control_effort)

    env.close()

    safeties = np.asarray(safeties)
    speeds = np.asarray(speeds)
    target_errors = np.asarray(target_errors)
    control_efforts = np.asarray(control_efforts)

    return {
        # Safety
        "mean_safety": np.mean(safeties),
        "std_safety": np.std(safeties),
        "min_safety": np.min(safeties),
        # Touchdown speed
        "mean_speed": np.mean(speeds),
        "std_speed": np.std(speeds),
        "max_speed": np.max(speeds),
        # Target error
        "mean_target_error": np.mean(target_errors),
        "std_target_error": np.std(target_errors),
        # Control effort
        "mean_control_effort": np.mean(control_efforts),
        "std_control_effort": np.std(control_efforts),
    }


# ============================================================
# Generic robustness sweep
# ============================================================


def run_robustness_sweep(
    model,
    parameter_name,
    values,
):
    results = []

    print(f"\nRobustness sweep: {parameter_name}")
    print("-" * 60)

    for value in values:
        print(f"Evaluating {parameter_name} = {value}")

        env_params = {parameter_name: value}

        metrics = evaluate_fixed_policy(
            model,
            env_params,
        )

        results.append(
            {
                "value": value,
                **metrics,
            }
        )

        print(
            f"  safety = {metrics['mean_safety']:.3f} ± {metrics['std_safety']:.3f}, min = {metrics['min_safety']:.3f}"
        )

        print(
            f"  touchdown speed = "
            f"{metrics['mean_speed']:.3f} "
            f"± {metrics['std_speed']:.3f} m/s"
            f", max = {metrics['max_speed']:.3f} m/s"
        )

        print(f"  target error = {metrics['mean_target_error']:.2f} ± {metrics['std_target_error']:.2f} m")

        print(f"  control effort = {metrics['mean_control_effort']:.4f} ± {metrics['std_control_effort']:.4f}")

    return results


# ============================================================
# Save results
# ============================================================


def save_results(
    filename,
    results,
):
    np.savez(
        filename,
        values=np.array([r["value"] for r in results]),
        mean_safeties=np.array([r["mean_safety"] for r in results]),
        std_safeties=np.array([r["std_safety"] for r in results]),
        min_safeties=np.array([r["min_safety"] for r in results]),
        mean_speeds=np.array([r["mean_speed"] for r in results]),
        std_speeds=np.array([r["std_speed"] for r in results]),
        max_speeds=np.array([r["max_speed"] for r in results]),
        mean_target_errors=np.array([r["mean_target_error"] for r in results]),
        std_target_errors=np.array([r["std_target_error"] for r in results]),
        mean_control_efforts=np.array([r["mean_control_effort"] for r in results]),
        std_control_efforts=np.array([r["std_control_effort"] for r in results]),
    )


# ============================================================
# Plot robustness analysis
# ============================================================


def plot_robustness(
    downrange_results,
    acceleration_results,
    descent_results,
):
    """
    Top row:
        touchdown safety mean ± std and minimum value

    Bottom row:
        touchdown speed mean ± std and maximum value
    """

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13, 8),
        sharex="col",
    )

    experiments = [
        (
            downrange_results,
            "Initial downrange velocity [m/s]",
            NOMINAL_DOWNRANGE_VELOCITY,
            "Downrange velocity",
        ),
        (
            acceleration_results,
            r"Maximum acceleration [m/s$^2$]",
            NOMINAL_MAX_ACCELERATION,
            "Control authority",
        ),
        (
            descent_results,
            "Descent time [s]",
            NOMINAL_DESCENT_TIME,
            "Descent time",
        ),
    ]

    for col, (
        results,
        xlabel,
        nominal_value,
        title,
    ) in enumerate(experiments):
        values = np.array([r["value"] for r in results])

        # ====================================================
        # Top row: safety
        # ====================================================

        mean_safety = np.array([r["mean_safety"] for r in results])

        std_safety = np.array([r["std_safety"] for r in results])

        min_safety = np.array([r["min_safety"] for r in results])

        ax_safety = axes[0, col]

        ax_safety.errorbar(
            values,
            mean_safety,
            yerr=std_safety,
            marker="o",
            capsize=4,
            label="Mean ± std",
        )

        ax_safety.plot(
            values,
            min_safety,
            marker="s",
            color="r",
            linestyle="--",
            label="Minimum",
        )

        ax_safety.axvline(
            nominal_value,
            linestyle="-",
            color="k",
            label="Training condition",
        )

        ax_safety.set_ylim(0, 1)
        ax_safety.set_title(title)
        ax_safety.grid(True)

        if col == 0:
            ax_safety.set_ylabel("Touchdown safety")

        # ====================================================
        # Bottom row: touchdown speed
        # ====================================================

        mean_speed = np.array([r["mean_speed"] for r in results])

        std_speed = np.array([r["std_speed"] for r in results])

        max_speed = np.array([r["max_speed"] for r in results])

        ax_speed = axes[1, col]

        ax_speed.errorbar(
            values,
            mean_speed,
            yerr=std_speed,
            marker="o",
            capsize=4,
            label="Mean ± std",
        )

        ax_speed.plot(
            values,
            max_speed,
            marker="s",
            color="r",
            linestyle="--",
            label="Maximum",
        )

        ax_speed.axvline(
            nominal_value,
            linestyle="-",
            color="k",
            label="Training condition",
        )

        ax_speed.set_xlabel(xlabel)

        ax_speed.grid(True)

        if col == 0:
            ax_speed.set_ylabel("Touchdown speed [m/s]")

    # ========================================================
    # Shared legend
    # ========================================================

    safety_handles, safety_labels = axes[0, 0].get_legend_handles_labels()

    speed_handles, speed_labels = axes[1, 0].get_legend_handles_labels()

    # Use descriptive shared labels that apply to both rows
    handles = [
        safety_handles[2],
        safety_handles[0],
        safety_handles[1],
    ]

    labels = [
        "Mean ± std",
        "Worst case (min safety / max speed)",
        "Training condition",
    ]

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    return fig, axes


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print(f"Loading fixed PPO model: {MODEL_PATH}")

    model = PPO.load(MODEL_PATH)

    # --------------------------------------------------------
    # Initial downrange velocity
    # --------------------------------------------------------

    downrange_results = run_robustness_sweep(
        model,
        "nominal_downrange_velocity",
        DOWNRANGE_VELOCITIES,
    )

    save_results(
        "robustness_downrange_velocity.npz",
        downrange_results,
    )

    # --------------------------------------------------------
    # Maximum acceleration
    # --------------------------------------------------------

    acceleration_results = run_robustness_sweep(
        model,
        "max_acceleration",
        MAX_ACCELERATIONS,
    )

    save_results(
        "robustness_max_acceleration.npz",
        acceleration_results,
    )

    # --------------------------------------------------------
    # Descent time
    # --------------------------------------------------------

    descent_results = run_robustness_sweep(
        model,
        "descent_time",
        DESCENT_TIMES,
    )

    save_results(
        "robustness_descent_time.npz",
        descent_results,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plot_robustness(
        downrange_results,
        acceleration_results,
        descent_results,
    )

    fig.savefig("ppo_environment_robustness.pdf", bbox_inches="tight", dpi=300)

    plt.show()

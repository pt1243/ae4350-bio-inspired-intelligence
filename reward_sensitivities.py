from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import LunarHazardEnvironment


# ============================================================
# Configuration
# ============================================================

TRAINING_STEPS = 100_000
N_EVAL_EPISODES = 100

TRAINING_SEED = 0
EVAL_SEED_START = 40_000


# ------------------------------------------------------------
# Nominal reward parameters
# ------------------------------------------------------------

NOMINAL_REWARDS = {
    "control_weight": 5.0,
    "velocity_weight": 20.0,
    "safety_weight": 100.0,
    "target_weight": 20.0,
}


# ------------------------------------------------------------
# Reward-weight sensitivity ranges
# ------------------------------------------------------------

REWARD_SWEEPS = {
    "control_weight": [
        0.5,
        1.5,
        5.0,
        15.0,
        50.0,
    ],
    "velocity_weight": [
        2.0,
        6.0,
        20.0,
        60.0,
        200.0,
    ],
    "safety_weight": [
        10.0,
        30.0,
        100.0,
        300.0,
        1000.0,
    ],
    "target_weight": [
        2.0,
        6.0,
        20.0,
        60.0,
        200.0,
    ],
}


# ------------------------------------------------------------
# Nominal PPO settings
# ------------------------------------------------------------

MODEL_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.999,
    "gae_lambda": 0.99,
    "clip_range": 0.2,
}


# ============================================================
# Evaluation
# ============================================================


def evaluate_model(
    model,
    env_params,
    n_episodes=N_EVAL_EPISODES,
):
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
        "mean_safety": np.mean(safeties),
        "std_safety": np.std(safeties),
        "min_safety": np.min(safeties),
        "mean_speed": np.mean(speeds),
        "std_speed": np.std(speeds),
        "max_speed": np.max(speeds),
        "mean_target_error": np.mean(target_errors),
        "std_target_error": np.std(target_errors),
        "mean_control_effort": np.mean(control_efforts),
        "std_control_effort": np.std(control_efforts),
    }


# ============================================================
# File naming
# ============================================================


def format_value(value):
    """
    Convert numerical weight into a convenient filename string.
    """

    return f"{value:g}".replace(".", "p")


def get_result_filename(
    parameter_name,
    value,
):
    value_string = format_value(value)

    return Path(f"reward_sensitivity_{parameter_name}_{value_string}.npz")


def get_model_filename(
    parameter_name,
    value,
):
    value_string = format_value(value)

    return f"reward_sensitivity_{parameter_name}_{value_string}"


# ============================================================
# Run one reward configuration
# ============================================================


def run_configuration(
    parameter_name,
    value,
):
    result_path = get_result_filename(
        parameter_name,
        value,
    )

    # --------------------------------------------------------
    # Existing result
    # --------------------------------------------------------

    if result_path.exists():
        print(f"Loading existing result: {result_path}")

        data = np.load(result_path)

        return {
            "value": value,
            "mean_safety": float(data["mean_safety"]),
            "std_safety": float(data["std_safety"]),
            "min_safety": float(data["min_safety"]),
            "mean_speed": float(data["mean_speed"]),
            "std_speed": float(data["std_speed"]),
            "max_speed": float(data["max_speed"]),
            "mean_target_error": float(data["mean_target_error"]),
            "std_target_error": float(data["std_target_error"]),
            "mean_control_effort": float(data["mean_control_effort"]),
            "std_control_effort": float(data["std_control_effort"]),
        }

    # --------------------------------------------------------
    # Environment parameters
    # --------------------------------------------------------

    env_params = NOMINAL_REWARDS.copy()

    env_params[parameter_name] = value

    env = LunarHazardEnvironment(**env_params)

    env = Monitor(env)

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=TRAINING_SEED,
        verbose=0,
        **MODEL_PARAMS,
    )

    print(f"\nTraining {parameter_name} = {value:g}")

    model.learn(
        total_timesteps=TRAINING_STEPS,
        progress_bar=True,
    )

    model_name = get_model_filename(
        parameter_name,
        value,
    )

    model.save(model_name)

    env.close()

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        env_params,
    )

    print(f"  safety = {metrics['mean_safety']:.3f} ± {metrics['std_safety']:.3f} (min {metrics['min_safety']:.3f})")

    print(
        f"  touchdown speed = "
        f"{metrics['mean_speed']:.3f} "
        f"± {metrics['std_speed']:.3f} m/s "
        f"(max {metrics['max_speed']:.3f})"
    )

    print(f"  target error = {metrics['mean_target_error']:.2f} ± {metrics['std_target_error']:.2f} m")

    print(f"  control effort = {metrics['mean_control_effort']:.4f} ± {metrics['std_control_effort']:.4f}")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.savez(
        result_path,
        parameter_name=parameter_name,
        value=value,
        **metrics,
    )

    return {
        "value": value,
        **metrics,
    }


# ============================================================
# Run complete sweep
# ============================================================


def run_reward_sensitivity():

    all_results = {}

    for (
        parameter_name,
        values,
    ) in REWARD_SWEEPS.items():
        print("\n")
        print("=" * 70)
        print(f"SENSITIVITY: {parameter_name}")
        print("=" * 70)

        results = []

        for value in values:
            result = run_configuration(
                parameter_name,
                value,
            )

            results.append(result)

        all_results[parameter_name] = results

    return all_results


# ============================================================
# Plot safety sensitivity
# ============================================================


def plot_reward_sensitivity(
    all_results,
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 8),
    )

    axes = axes.flatten()

    plot_settings = [
        (
            "control_weight",
            "Control-effort weight",
        ),
        (
            "velocity_weight",
            "Touchdown-velocity weight",
        ),
        (
            "safety_weight",
            "Landing-safety weight",
        ),
        (
            "target_weight",
            "Target-error weight",
        ),
    ]

    for ax, (
        parameter_name,
        title,
    ) in zip(
        axes,
        plot_settings,
    ):
        results = all_results[parameter_name]

        values = np.array([r["value"] for r in results])

        mean_safety = np.array([r["mean_safety"] for r in results])

        std_safety = np.array([r["std_safety"] for r in results])

        min_safety = np.array([r["min_safety"] for r in results])

        # Mean ± standard deviation
        ax.errorbar(
            values,
            mean_safety,
            yerr=std_safety,
            marker="o",
            capsize=4,
            label="Mean ± std",
        )

        # Minimum safety
        ax.plot(
            values,
            min_safety,
            marker="s",
            linestyle="--",
            label="Minimum safety",
        )

        # Nominal reward value
        nominal_value = NOMINAL_REWARDS[parameter_name]

        ax.axvline(
            nominal_value,
            linestyle=":",
            label="Nominal value",
        )

        # Log scale because sweep spans
        # approximately two orders of magnitude.
        ax.set_xscale("log")

        ax.set_xlabel("Reward weight")

        ax.set_ylabel("Touchdown safety")

        ax.set_ylim(
            0,
            1,
        )

        ax.set_title(title)

        ax.grid(
            True,
            which="both",
        )

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

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
# Print summary
# ============================================================


def print_summary(
    all_results,
):

    print("\n")
    print("=" * 90)
    print("REWARD-WEIGHT SENSITIVITY SUMMARY")
    print("=" * 90)

    for parameter_name, results in all_results.items():
        print(f"\n{parameter_name}")

        for result in results:
            print(
                f"  {result['value']:8g} | "
                f"safety "
                f"{result['mean_safety']:.3f} "
                f"± {result['std_safety']:.3f} | "
                f"min "
                f"{result['min_safety']:.3f} | "
                f"speed "
                f"{result['mean_speed']:.3f} m/s | "
                f"error "
                f"{result['mean_target_error']:.1f} m | "
                f"effort "
                f"{result['mean_control_effort']:.4f}"
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = run_reward_sensitivity()

    print_summary(results)

    fig, axes = plot_reward_sensitivity(results)

    fig.savefig(
        "ppo_reward_weight_sensitivity.pdf",
        bbox_inches="tight",
    )

    plt.show()

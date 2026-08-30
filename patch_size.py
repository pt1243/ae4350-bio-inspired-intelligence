from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import LunarHazardEnvironment


# ============================================================
# Configuration
# ============================================================

PATCH_SIZES = [
    3,
    5,
    7,  # nominal
    9,
    11,
]

NOMINAL_PATCH_SIZE = 7

TRAINING_STEPS = 100_000

N_EVAL_EPISODES = 100
EVAL_SEED_START = 30_000

TRAINING_SEED = 0


# Final reward parameters selected from the reward search
BASE_ENV_PARAMS = {
    "control_weight": 5.0,
    "velocity_weight": 20.0,
    "safety_weight": 100.0,
    "target_weight": 20.0,
}


# Nominal PPO parameters
BASE_MODEL_PARAMS = {
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
    patch_size,
    n_episodes=N_EVAL_EPISODES,
):
    """
    Evaluate a trained model using the same patch size
    with which it was trained.
    """

    env_params = BASE_ENV_PARAMS.copy()
    env_params["patch_size"] = patch_size

    env = LunarHazardEnvironment(**env_params)

    safeties = []
    speeds = []
    target_errors = []
    control_efforts = []
    min_safety = np.inf
    max_speed = 0

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
        min_safety = min(min_safety, info["touchdown_safety"])

        speeds.append(info["touchdown_speed"])
        max_speed = max(max_speed, info["touchdown_speed"])

        target_errors.append(info["target_error"])

        control_efforts.append(control_effort)

    env.close()

    return {
        "mean_safety": np.mean(safeties),
        "std_safety": np.std(safeties),
        "mean_speed": np.mean(speeds),
        "std_speed": np.std(speeds),
        "mean_target_error": np.mean(target_errors),
        "std_target_error": np.std(target_errors),
        "mean_control_effort": np.mean(control_efforts),
        "std_control_effort": np.std(control_efforts),
        "min_safety": min_safety,
        "max_speed": max_speed,
    }


# ============================================================
# Train one patch-size model
# ============================================================


def train_patch_model(
    patch_size,
):
    """
    Train a PPO model for one hazard-map patch size.
    """

    env_params = BASE_ENV_PARAMS.copy()
    env_params["patch_size"] = patch_size

    env = LunarHazardEnvironment(**env_params)

    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        seed=TRAINING_SEED,
        verbose=0,
        **BASE_MODEL_PARAMS,
    )

    print(f"Training patch size {patch_size}x{patch_size}...")

    model.learn(
        total_timesteps=TRAINING_STEPS,
        progress_bar=True,
    )

    env.close()

    return model


# ============================================================
# Run one patch-size case
# ============================================================


def run_patch_size(
    patch_size,
):
    """
    Load existing results if available.
    Otherwise train and evaluate a new model.
    """

    name = f"patch_size_{patch_size}x{patch_size}"

    model_path = Path(name)
    result_path = Path(f"{name}_results.npz")

    # --------------------------------------------------------
    # Use existing data
    # --------------------------------------------------------

    if result_path.exists():
        print(f"Found existing results for {patch_size}x{patch_size}. Skipping training.")

        data = np.load(result_path)

        return {
            "patch_size": patch_size,
            "mean_safety": float(data["mean_safety"]),
            "std_safety": float(data["std_safety"]),
            "mean_speed": float(data["mean_speed"]),
            "std_speed": float(data["std_speed"]),
            "mean_target_error": float(data["mean_target_error"]),
            "std_target_error": float(data["std_target_error"]),
            "mean_control_effort": float(data["mean_control_effort"]),
            "std_control_effort": float(data["std_control_effort"]),
            "min_safety": float(data["min_safety"]),
            "max_speed": float(data["max_speed"]),
        }

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_patch_model(patch_size)

    model.save(str(model_path))

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        patch_size,
    )

    print(f"\nPatch {patch_size}x{patch_size}:")

    print(f"  Safety: {metrics['mean_safety']:.3f} ± {metrics['std_safety']:.3f}")

    print(f"  Minimum safety:  {metrics['min_safety']:.3f}")

    print(f"  Touchdown speed: {metrics['mean_speed']:.3f} ± {metrics['std_speed']:.3f} m/s")

    print(f"  Maximum speed:  {metrics['max_speed']:.3f}")

    print(f"  Target error: {metrics['mean_target_error']:.2f} ± {metrics['std_target_error']:.2f} m")

    print(f"  Control effort: {metrics['mean_control_effort']:.4f} ± {metrics['std_control_effort']:.4f}")

    # --------------------------------------------------------
    # Save evaluation results
    # --------------------------------------------------------

    np.savez(
        result_path,
        patch_size=patch_size,
        mean_safety=metrics["mean_safety"],
        std_safety=metrics["std_safety"],
        mean_speed=metrics["mean_speed"],
        std_speed=metrics["std_speed"],
        mean_target_error=metrics["mean_target_error"],
        std_target_error=metrics["std_target_error"],
        mean_control_effort=metrics["mean_control_effort"],
        std_control_effort=metrics["std_control_effort"],
        min_safety=metrics["min_safety"],
        max_speed=metrics["max_speed"],
    )

    return {
        "patch_size": patch_size,
        **metrics,
    }


# ============================================================
# Plot
# ============================================================


def plot_patch_size_sensitivity(
    results,
):
    """
    Compare:
        1. touchdown safety
        2. touchdown speed
        3. target error

    for each patch size.
    """

    patch_sizes = np.array([r["patch_size"] for r in results])

    mean_safeties = np.array([r["mean_safety"] for r in results])

    std_safeties = np.array([r["std_safety"] for r in results])

    mean_speeds = np.array([r["mean_speed"] for r in results])

    std_speeds = np.array([r["std_speed"] for r in results])

    mean_errors = np.array([r["mean_target_error"] for r in results])

    std_errors = np.array([r["std_target_error"] for r in results])

    min_safeties = np.array([r["min_safety"] for r in results])

    max_speeds = np.array([r["max_speed"] for r in results])

    # ========================================================
    # Figure
    # ========================================================

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(10, 4),
    )

    # --------------------------------------------------------
    # Safety
    # --------------------------------------------------------

    axes[0].errorbar(
        patch_sizes,
        mean_safeties,
        yerr=std_safeties,
        marker="o",
        capsize=4,
        label=r"Mean ± 1$\sigma$",
    )

    axes[0].plot(patch_sizes, min_safeties, marker="o", linestyle="--", c="tab:red", label="Minimum safety")

    axes[0].axvline(NOMINAL_PATCH_SIZE, linestyle="--", label="Nominal patch size", c="k")

    axes[0].set_xlabel("Patch size")

    axes[0].set_ylabel("Touchdown safety [-]")

    axes[0].set_ylim(
        0,
        1,
    )

    axes[0].set_title("Landing safety")

    axes[0].grid(True)

    # --------------------------------------------------------
    # Touchdown speed
    # --------------------------------------------------------

    axes[1].errorbar(
        patch_sizes,
        mean_speeds,
        yerr=std_speeds,
        marker="o",
        capsize=4,
    )

    axes[1].plot(patch_sizes, max_speeds, marker="o", linestyle="--", c="tab:orange", label="Maximum speed")

    axes[1].axvline(NOMINAL_PATCH_SIZE, linestyle="--", c="k")

    axes[1].set_xlabel("Patch size")

    axes[1].set_ylabel("Touchdown speed [m/s]")

    axes[1].set_title("Touchdown velocity")

    axes[1].grid(True)

    # --------------------------------------------------------
    # Target error
    # --------------------------------------------------------

    axes[2].errorbar(
        patch_sizes,
        mean_errors,
        yerr=std_errors,
        marker="o",
        capsize=4,
    )

    axes[2].axvline(NOMINAL_PATCH_SIZE, linestyle="--", c="k")

    axes[2].set_xlabel("Patch size")

    axes[2].set_ylabel("Target error [m]")

    axes[2].set_title("Landing site deviation")

    axes[2].grid(True)

    # --------------------------------------------------------
    # Integer patch-size ticks
    # --------------------------------------------------------

    for ax in axes:
        ax.set_xticks(patch_sizes)

        ax.set_xticklabels([f"{size}x{size}" for size in patch_sizes])

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()
    handles_ax1, labels_ax1 = axes[1].get_legend_handles_labels()
    handles += handles_ax1
    labels += labels_ax1

    # reorder
    handles = [handles[2], handles[1], handles[0], handles[3]]
    labels = [labels[2], labels[1], labels[0], labels[3]]

    # breakpoint()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    return fig, axes


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    results = []

    for patch_size in PATCH_SIZES:
        result = run_patch_size(patch_size)

        results.append(result)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("PATCH SIZE SENSITIVITY SUMMARY")
    print("=" * 75)

    for result in results:
        print(
            f"{result['patch_size']}x"
            f"{result['patch_size']} | "
            f"safety = "
            f"{result['mean_safety']:.3f} "
            f"± {result['std_safety']:.3f} | "
            f"speed = "
            f"{result['mean_speed']:.3f} "
            f"± {result['std_speed']:.3f} m/s | "
            f"target error = "
            f"{result['mean_target_error']:.2f} "
            f"± {result['std_target_error']:.2f} m"
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plot_patch_size_sensitivity(results)

    fig.savefig("plots/ppo_patch_size_sensitivity.pdf", bbox_inches="tight", dpi=300)

    # plt.show()

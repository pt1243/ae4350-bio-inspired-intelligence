from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from train import train_with_parameters

NOMINAL_PARAMS = {
    "learning_rate": 3e-4,
    "gamma": 0.999,
    "gae_lambda": 0.99,
    "clip_range": 0.2,
}


# ============================================================
# Parameter ranges
# ============================================================

LEARNING_RATES = [1e-5, 1e-4, 3e-4, 1e-3, 1e-2]
GAMMAS = [0.95, 0.995, 0.999, 0.9995]
CLIP_RANGES = [0.05, 0.1, 0.2, 0.3, 0.5]
NETWORK_SIZES = [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256]]


# ============================================================
# Helpers
# ============================================================

# def value_to_filename(value) -> str:
#     """
#     Convert a parameter value into a filesystem-friendly string.
#     """
#     if isinstance(value, float):
#         return str(value).replace(".", "p")

#     return str(value)


def train_if_needed(
    model_name: str,
    model_params: dict,
):
    data_path = Path(f"learning_curve_{model_name}.npz")

    if data_path.exists():
        print(f"Trained data for {model_name} exists, continuing...")
        return data_path

    train_with_parameters(
        filename=model_name,
        model_params=model_params,
    )

    return data_path


def get_final_safety(data_path):
    """
    Return the final recorded evaluation safety mean and std.
    """

    data = np.load(data_path)

    mean_safety = data["mean_safeties"][-1]
    std_safety = data["std_safeties"][-1]

    return mean_safety, std_safety


# ============================================================
# Sensitivity sweeps
# ============================================================


def run_learning_rate_sweep():
    results = []

    for lr in LEARNING_RATES:
        params = NOMINAL_PARAMS.copy()
        params["learning_rate"] = lr

        name = f"sensitivity_learning_rate_{lr:.0e}"

        data_path = train_if_needed(name, params)

        mean, std = get_final_safety(data_path)

        results.append((lr, mean, std))

    return results


def run_gamma_sweep():
    results = []

    for gamma in GAMMAS:
        params = NOMINAL_PARAMS.copy()
        params["gamma"] = gamma

        value_name = str(gamma).replace(".", "p")

        name = f"sensitivity_gamma_{value_name}"

        data_path = train_if_needed(name, params)

        mean, std = get_final_safety(data_path)

        results.append((gamma, mean, std))

    return results


def run_clip_range_sweep():
    results = []

    for clip_range in CLIP_RANGES:
        params = NOMINAL_PARAMS.copy()
        params["clip_range"] = clip_range

        value_name = str(clip_range).replace(".", "p")

        name = f"sensitivity_clip_range_{value_name}"

        data_path = train_if_needed(name, params)

        mean, std = get_final_safety(data_path)

        results.append((clip_range, mean, std))

    return results


def run_network_sweep():
    results = []

    for network in NETWORK_SIZES:
        params = NOMINAL_PARAMS.copy()

        params["policy_kwargs"] = {
            "net_arch": {
                "pi": network,
                "vf": network,
            }
        }

        network_name = "x".join(str(n) for n in network)
        name = f"sensitivity_network_{network_name}"

        data_path = train_if_needed(name, params)

        mean, std = get_final_safety(data_path)

        results.append((network_name, mean, std))

    return results


# ============================================================
# Plotting
# ============================================================


def plot_sensitivity(
    learning_rate_results,
    gamma_results,
    clip_results,
    network_results,
):
    fig, axes = plt.subplots(4, 1, figsize=(8, 13))

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    lr_values = np.array([r[0] for r in learning_rate_results])

    lr_means = np.array([r[1] for r in learning_rate_results])

    lr_stds = np.array([r[2] for r in learning_rate_results])

    axes[0].errorbar(lr_values, lr_means, yerr=lr_stds, marker="o", capsize=4)

    axes[0].set_xscale("log")
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("Touchdown safety")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Learning-rate sensitivity")
    axes[0].grid(True)

    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    gamma_values = np.array([r[0] for r in gamma_results])

    gamma_means = np.array([r[1] for r in gamma_results])

    gamma_stds = np.array([r[2] for r in gamma_results])

    axes[1].errorbar(gamma_values, gamma_means, yerr=gamma_stds, marker="o", capsize=4)

    axes[1].set_xlabel(r"Discount factor $\gamma$")
    axes[1].set_ylabel("Touchdown safety")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Discount-factor sensitivity")
    axes[1].grid(True)

    # --------------------------------------------------------
    # Clip range
    # --------------------------------------------------------

    clip_values = np.array([r[0] for r in clip_results])

    clip_means = np.array([r[1] for r in clip_results])

    clip_stds = np.array([r[2] for r in clip_results])

    axes[2].errorbar(clip_values, clip_means, yerr=clip_stds, marker="o", capsize=4)

    axes[2].set_xlabel("PPO clip range")
    axes[2].set_ylabel("Touchdown safety")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Clip-range sensitivity")
    axes[2].grid(True)

    # --------------------------------------------------------
    # Network architecture
    # --------------------------------------------------------

    network_labels = [r[0] for r in network_results]

    network_means = np.array([r[1] for r in network_results])

    network_stds = np.array([r[2] for r in network_results])

    x = np.arange(len(network_labels))

    axes[3].errorbar(x, network_means, yerr=network_stds, marker="o", capsize=4)

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(network_labels)

    axes[3].set_xlabel("Actor/critic hidden layers")
    axes[3].set_ylabel("Touchdown safety")
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Network-size sensitivity")
    axes[3].grid(True)

    fig.tight_layout()

    return fig, axes

if __name__ == "__main__":

    lr_results = (
        run_learning_rate_sweep()
    )

    gamma_results = (
        run_gamma_sweep()
    )

    clip_results = (
        run_clip_range_sweep()
    )

    network_results = (
        run_network_sweep()
    )

    fig, axes = plot_sensitivity(
        lr_results,
        gamma_results,
        clip_results,
        network_results,
    )

    fig.savefig(
        "ppo_hyperparameter_sensitivity.pdf",
        bbox_inches="tight",
    )

    plt.show()
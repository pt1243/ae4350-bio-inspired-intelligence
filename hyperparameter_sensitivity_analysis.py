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

LEARNING_RATES = [1e-5, 1e-4, 3e-4, 1e-3, 5e-3]
GAMMAS = [0.8, 0.99, 0.999, 0.9995]
CLIP_RANGES = [0.05, 0.1, 0.2, 0.3, 0.5]
NETWORK_SIZES = [[8, 8], [16, 16], [32, 32], [64, 64], [128, 128]]


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


def train_if_needed(model_name: str, model_params: dict, record_clipping: bool = False):
    data_path = Path(f"learning_curve_{model_name}.npz")

    if data_path.exists():
        print(f"Trained data for {model_name} exists, continuing...")
        return data_path

    train_with_parameters(filename=model_name, model_params=model_params, record_clipping=record_clipping)

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

        # Load training diagnostics
        diagnostics_path = Path(f"ppo_diagnostics_{name}.npz")

        mean_clip_fraction = np.nan

        if diagnostics_path.exists():
            diagnostics = np.load(diagnostics_path)

            clip_fractions = diagnostics["clip_fraction"]

            mean_clip_fraction = np.mean(clip_fractions)

            print(f"clip_range = {clip_range:.3f} -> mean clip fraction = {mean_clip_fraction:.4f}")

        else:
            print(f"clip_range = {clip_range:.3f} -> diagnostics file not found")

        results.append(
            (
                clip_range,
                mean,
                std,
                mean_clip_fraction,
            )
        )

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
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6))

    axes = axes.flatten()

    nominal_learning_rate = 3e-4
    nominal_gamma = 0.999
    nominal_clip = 0.2
    nominal_network = "64x64"

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    lr_values = np.array([r[0] for r in learning_rate_results])

    lr_means = np.array([r[1] for r in learning_rate_results])

    lr_stds = np.array([r[2] for r in learning_rate_results])

    axes[0].errorbar(
        lr_values,
        lr_means,
        yerr=lr_stds,
        marker="o",
        capsize=4,
        label=r"Mean ± 1$\sigma$",
    )

    axes[0].axvline(nominal_learning_rate, linestyle="--", label="Nominal value", c="k")

    axes[0].set_xscale("log")

    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("Touchdown safety")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Learning rate sensitivity")
    axes[0].grid(True)

    # --------------------------------------------------------
    # Gamma / discount-factor sensitivity
    # --------------------------------------------------------

    gamma_values = np.array([r[0] for r in gamma_results])

    gamma_means = np.array([r[1] for r in gamma_results])

    gamma_stds = np.array([r[2] for r in gamma_results])

    # Plot logarithmically in terms of (1 - gamma)
    discount_rates = 1.0 - gamma_values
    nominal_discount_rate = 1.0 - nominal_gamma

    axes[1].errorbar(
        discount_rates,
        gamma_means,
        yerr=gamma_stds,
        marker="o",
        capsize=4,
    )

    axes[1].axvline(nominal_discount_rate, linestyle="--", c="k")

    axes[1].set_xscale("log")

    # Show gamma values as the tick labels
    axes[1].set_xticks(discount_rates)
    axes[1].set_xticklabels([f"{gamma:g}" for gamma in gamma_values])

    # Reverse so gamma increases from left to right
    axes[1].invert_xaxis()

    axes[1].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel("Touchdown safety")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Discount factor sensitivity")
    axes[1].grid(True)

    # --------------------------------------------------------
    # Clip range
    # --------------------------------------------------------

    clip_values = np.array([r[0] for r in clip_results])

    clip_means = np.array([r[1] for r in clip_results])

    clip_stds = np.array([r[2] for r in clip_results])

    axes[2].errorbar(
        clip_values,
        clip_means,
        yerr=clip_stds,
        marker="o",
        capsize=4,
    )

    axes[2].axvline(nominal_clip, linestyle="--", c="k")

    axes[2].set_xlabel(r"$\epsilon$")
    axes[2].set_ylabel("Touchdown safety")
    axes[2].set_ylim(0, 1)
    axes[2].set_title("Clip range sensitivity")
    axes[2].grid(True)

    # --------------------------------------------------------
    # Network size
    # --------------------------------------------------------

    network_labels = [r[0] for r in network_results]

    network_means = np.array([r[1] for r in network_results])

    network_stds = np.array([r[2] for r in network_results])

    x = np.arange(len(network_labels))

    axes[3].errorbar(
        x,
        network_means,
        yerr=network_stds,
        marker="o",
        capsize=4,
    )

    nominal_network_index = network_labels.index(nominal_network)

    axes[3].axvline(nominal_network_index, linestyle="--", c="k")

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(network_labels)

    axes[3].set_xlabel("Actor and critic hidden layers")
    axes[3].set_ylabel("Touchdown safety")
    axes[3].set_ylim(0, 1)
    axes[3].set_title("Neural network size sensitivity")
    axes[3].grid(True)

    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        reversed(handles),
        reversed(labels),
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fig.savefig("plots/ppo_hyperparameter_sensitivity.pdf", bbox_inches="tight", dpi=300)

    return fig, axes


def plot_learning_rate_histories():
    learning_rates = LEARNING_RATES

    fig, ax = plt.subplots(figsize=(5, 3))

    for lr in learning_rates:
        model_name = f"sensitivity_learning_rate_{lr:.0e}"

        data_path = Path(f"learning_curve_{model_name}.npz")

        if not data_path.exists():
            print(f"Skipping {lr:g}: {data_path} not found.")
            continue

        data = np.load(data_path)

        steps = data["steps"]
        mean_returns = data["mean_returns"]

        if lr == 3e-4:
            label = f"{lr:e} (nominal)"
        else:
            label = f"{lr:e}"

        ax.plot(
            steps,
            mean_returns,
            marker="o",
            label=label,
        )

    ax.set_xlabel("Training steps [-]")
    ax.set_ylabel("Mean total reward [-]")
    # ax.set_title("Effect of learning rate on PPO training")

    ax.grid(True)
    fig.legend(
        labels=["1e-5", "1e-4", "3e-4 (nominal)", "1e-3", "5e-3"], loc="outside right center", bbox_to_anchor=[1.3, 0.5]
    )

    fig.tight_layout()
    fig.savefig("plots/learning_rate_training_histories.pdf", bbox_inches="tight", dpi=300)

    return fig, ax


if __name__ == "__main__":
    lr_results = run_learning_rate_sweep()

    gamma_results = run_gamma_sweep()

    clip_results = run_clip_range_sweep()

    network_results = run_network_sweep()

    plot_sensitivity(lr_results, gamma_results, clip_results, network_results)
    plot_learning_rate_histories()

    # plt.show()

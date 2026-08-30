import csv
import itertools

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import LunarHazardEnvironment

# ------------------------------------------------------------
# Search settings
# ------------------------------------------------------------

CONTROL_WEIGHTS = [0.1, 1.0, 5.0]
VELOCITY_WEIGHTS = [5.0, 20.0, 50.0]
TARGET_WEIGHTS = [1.0, 5.0, 20.0]

SAFETY_WEIGHT = 100.0

TRAINING_STEPS = 100_000
N_EVAL_EPISODES = 50

TRAINING_SEED = 0
EVAL_SEED_START = 20_000


def evaluate_model(model, env_params, n_episodes=50):
    """
    Evaluate one trained model on a fixed set of maps/initial conditions.
    """

    env = LunarHazardEnvironment(**env_params)

    returns = []
    safeties = []
    speeds = []
    target_errors = []
    control_efforts = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=EVAL_SEED_START + episode)

        terminated = False
        truncated = False

        episode_return = 0.0
        control_effort = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            acceleration = action * env.max_acceleration

            control_effort += np.sum(acceleration**2) * env.dt

            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward

        returns.append(episode_return)
        safeties.append(info["touchdown_safety"])
        speeds.append(info["touchdown_speed"])
        target_errors.append(info["target_error"])
        control_efforts.append(control_effort)

    env.close()

    return {
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "mean_safety": np.mean(safeties),
        "std_safety": np.std(safeties),
        "mean_speed": np.mean(speeds),
        "mean_target_error": np.mean(target_errors),
        "mean_control_effort": np.mean(control_efforts),
    }


def run_grid_search():

    combinations = list(itertools.product(CONTROL_WEIGHTS, VELOCITY_WEIGHTS, TARGET_WEIGHTS))

    results = []

    print(f"Running {len(combinations)} reward configurations...")

    for i, (control_weight, velocity_weight, target_weight) in enumerate(combinations, start=1):
        env_params = {
            "control_weight": control_weight,
            "velocity_weight": velocity_weight,
            "safety_weight": SAFETY_WEIGHT,
            "target_weight": target_weight,
        }

        print(
            f"\n[{i}/{len(combinations)}] control={control_weight}, velocity={velocity_weight}, target={target_weight}"
        )

        env = Monitor(LunarHazardEnvironment(**env_params))

        model = PPO(policy="MlpPolicy", env=env, gamma=0.999, gae_lambda=0.99, seed=TRAINING_SEED, verbose=0)

        model.learn(total_timesteps=TRAINING_STEPS, progress_bar=True)

        env.close()

        metrics = evaluate_model(
            model,
            env_params,
            n_episodes=N_EVAL_EPISODES,
        )

        result = {
            "control_weight": control_weight,
            "velocity_weight": velocity_weight,
            "safety_weight": SAFETY_WEIGHT,
            "target_weight": target_weight,
            **metrics,
        }

        results.append(result)

        print(
            f"  safety = "
            f"{metrics['mean_safety']:.3f}"
            f", speed = "
            f"{metrics['mean_speed']:.3f} m/s"
            f", target error = "
            f"{metrics['mean_target_error']:.2f} m"
            f", effort = "
            f"{metrics['mean_control_effort']:.3f}"
        )

    return results


def save_results(results, filename="reward_grid_search.csv"):
    fieldnames = list(results[0].keys())

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    results = run_grid_search()

    save_results(results)

    print("\nSaved results to reward_grid_search.csv")

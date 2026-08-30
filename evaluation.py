import numpy as np
from stable_baselines3 import PPO

from environment import LunarHazardEnvironment


def evaluate_model(model_path: str, n_episodes: int = 100, seed_start: int = 1000):
    env = LunarHazardEnvironment()
    model = PPO.load(model_path)

    results = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed_start + episode)

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

        results.append(
            {
                "return": episode_return,
                "touchdown_safety": info["touchdown_safety"],
                "touchdown_speed": info["touchdown_speed"],
                "target_error": info["target_error"],
                "control_effort": control_effort,
            }
        )

    env.close()

    return results


def run_episode_with_trajectory(env, model, seed=None):
    """
    Run one deterministic PPO episode and record the physical position
    and predicted touchdown point at each timestep.
    """
    obs, info = env.reset(seed=seed)

    positions = []
    touchdown_predictions = []
    times = []
    actions = []

    terminated = False
    truncated = False
    episode_return = 0.0

    while not (terminated or truncated):
        # Store state before applying the next action
        positions.append(env.position.copy())

        touchdown_prediction = env.position + env.velocity * env.time_remaining
        touchdown_predictions.append(touchdown_prediction.copy())

        times.append(env.descent_time - env.time_remaining)

        action, _ = model.predict(obs, deterministic=True)

        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += reward

    # Add final touchdown state
    positions.append(env.position.copy())

    touchdown_predictions.append(env.position.copy())

    times.append(env.descent_time)

    return {
        "positions": np.asarray(positions),
        "touchdown_predictions": np.asarray(touchdown_predictions),
        "times": np.asarray(times),
        "actions": np.asarray(actions),
        "hazard_map": env.hazard_map.copy(),
        "return": episode_return,
        "info": info,
    }


def print_summary(results):
    returns = np.array([r["return"] for r in results])
    safety = np.array([r["touchdown_safety"] for r in results])
    speed = np.array([r["touchdown_speed"] for r in results])
    target_error = np.array([r["target_error"] for r in results])
    control_effort = np.array([r["control_effort"] for r in results])

    print("\nEvaluation summary")
    print("------------------")
    print(f"Return:          {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"Safety:          {safety.mean():.3f} ± {safety.std():.3f}")
    print(f"Touchdown speed: {speed.mean():.3f} ± {speed.std():.3f} m/s")
    print(f"Target error:    {target_error.mean():.2f} ± {target_error.std():.2f} m")
    print(f"Control effort:  {control_effort.mean():.4f} ± {control_effort.std():.4f}")


if __name__ == "__main__":
    results = evaluate_model(model_path="base_model", n_episodes=100)
    print_summary(results)

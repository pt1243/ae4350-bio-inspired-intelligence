import numpy as np
from stable_baselines3 import PPO

from environment import LunarHazardEnvironment


def evaluate(n_episodes=10):
    env = LunarHazardEnvironment()

    model = PPO.load("ppo_lunar_test")

    returns = []

    for episode in range(n_episodes):
        obs, info = env.reset(seed=1000 + episode)

        terminated = False
        truncated = False
        episode_return = 0.0

        while not (terminated or truncated):

            action, _ = model.predict(
                obs,
                deterministic=True,
            )

            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward

        returns.append(episode_return)

        print(f"\nEpisode {episode + 1}")
        print(f"Return:         {episode_return:.2f}")
        print(f"Final position: {env.position}")
        print(f"Final velocity: {env.velocity}")
        print(
            f"Target error:   "
            f"{np.linalg.norm(env.position):.2f} m"
        )

    print("\n---------------------------")
    print(f"Mean return: {np.mean(returns):.2f}")
    print(f"Std return:  {np.std(returns):.2f}")


if __name__ == "__main__":
    evaluate()
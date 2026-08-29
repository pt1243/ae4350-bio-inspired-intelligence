import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from environment import LunarHazardEnvironment

def train():
    env = LunarHazardEnvironment()
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        gamma=0.999,
        gae_lambda=0.99,
        verbose=1,
        seed=0
    )

    model.learn(
        total_timesteps=200_000,
        progress_bar=True,
    )

    model.save("ppo_lunar_test")

    env.close()

if __name__ == "__main__":
    train()
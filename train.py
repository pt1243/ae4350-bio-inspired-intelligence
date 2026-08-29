import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from environment import LunarHazardEnvironment

class TrainingEvaluationCallback(BaseCallback):
    def __init__(self, eval_frequency=10_000, n_eval_episodes=50, eval_seed_start=10_000, verbose = 0):
        super().__init__(verbose)

        self.eval_frequency = eval_frequency
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed_start = eval_seed_start

        self.eval_steps = []
        self.mean_returns = []
        self.std_returns = []
        self.mean_safeties = []
        self.std_safeties = []
        self.mean_speeds = []
        self.mean_target_errors = []

    def _on_step(self):
        if self.num_timesteps % self.eval_frequency != 0:
            return True
        returns = []
        safeties = []
        speeds = []
        target_errors = []

        # separate env for evaluation
        eval_env = LunarHazardEnvironment()
        for episode in range(self.n_eval_episodes):
            obs, info = eval_env.reset(seed=self.eval_seed_start + episode)
            terminated = False
            truncated = False
            episode_return = 0.0
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_return += reward
            returns.append(episode_return)
            safeties.append(info["touchdown_safety"])
            speeds.append(info["touchdown_speed"])
            target_errors.append(info["target_error"])
        eval_env.close()

        self.eval_steps.append(self.num_timesteps)
        self.mean_returns.append(np.mean(returns))
        self.std_returns.append(np.std(returns))
        self.mean_safeties.append(np.mean(safeties))
        self.std_safeties.append(np.std(safeties))
        self.mean_speeds.append(np.mean(speeds))
        self.mean_target_errors.append(np.mean(target_errors))

        return True



def train(filename: str):
    env = LunarHazardEnvironment()
    env = Monitor(env)


    model = PPO(policy="MlpPolicy", env=env, gamma=0.999, gae_lambda=0.99, verbose=1, seed=0)

    eval_callback = TrainingEvaluationCallback(eval_frequency=10_000, n_eval_episodes=50, verbose=1)

    model.learn(
        total_timesteps=500_000,
        callback=eval_callback,
        progress_bar=True,
    )
    model.save(filename)
    env.close()

    np.savez(
        f"learning_curve_{filename}.npz",
        steps=eval_callback.eval_steps,
        mean_returns=eval_callback.mean_returns,
        std_returns=eval_callback.std_returns,
        mean_safeties=eval_callback.mean_safeties,
        std_safeties=eval_callback.std_safeties,
        mean_speeds=eval_callback.mean_speeds,
        mean_target_errors=eval_callback.mean_target_errors,
    )

if __name__ == "__main__":
    train(filename="ppo_lunar")

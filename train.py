from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from environment import LunarHazardEnvironment


class TrainingEvaluationCallback(BaseCallback):
    def __init__(self, eval_frequency=5_000, n_eval_episodes=50, eval_seed_start=10_000, verbose=0):
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
        self.std_speeds = []
        self.mean_target_errors = []
        self.std_target_errors = []
        self.mean_control_efforts = []
        self.std_control_efforts = []
        self.min_safeties = []

    def _on_step(self):
        if self.num_timesteps > 1 and self.num_timesteps % self.eval_frequency != 0:
            return True
        returns = []
        safeties = []
        speeds = []
        target_errors = []
        control_efforts = []

        # separate env for evaluation
        eval_env = LunarHazardEnvironment()
        for episode in range(self.n_eval_episodes):
            obs, info = eval_env.reset(seed=self.eval_seed_start + episode)
            terminated = False
            truncated = False
            episode_return = 0.0
            control_effort = 0.0
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                acceleration = action * eval_env.max_acceleration
                control_effort += np.sum(acceleration**2) * eval_env.dt
                obs, reward, terminated, truncated, info = eval_env.step(action)
                episode_return += reward
            returns.append(episode_return)
            safeties.append(info["touchdown_safety"])
            speeds.append(info["touchdown_speed"])
            target_errors.append(info["target_error"])
            control_efforts.append(control_effort)
        eval_env.close()

        self.eval_steps.append(self.num_timesteps)
        self.mean_returns.append(np.mean(returns))
        self.std_returns.append(np.std(returns))
        self.mean_safeties.append(np.mean(safeties))
        self.std_safeties.append(np.std(safeties))
        self.mean_speeds.append(np.mean(speeds))
        self.std_speeds.append(np.std(speeds))
        self.mean_target_errors.append(np.mean(target_errors))
        self.std_target_errors.append(np.std(target_errors))
        self.mean_control_efforts.append(np.mean(control_efforts))
        self.std_control_efforts.append(np.std(control_efforts))
        self.min_safeties.append(np.min(safeties))

        return True


class PPOTrainingDiagnosticsCallback(BaseCallback):
    def __init__(self):
        super().__init__()

        self.steps = []
        self.clip_fractions = []
        # self.approx_kls = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        logger_values = self.model.logger.name_to_value

        clip_fraction = logger_values.get("train/clip_fraction")

        # approx_kl = logger_values.get(
        #     "train/approx_kl"
        # )

        if clip_fraction is not None:
            self.steps.append(self.num_timesteps)

            self.clip_fractions.append(clip_fraction)

            # self.approx_kls.append(
            #     approx_kl
            # )


def train_with_parameters(
    filename: str,
    env_params: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    record_clipping: bool = False,
):
    if env_params is None:
        env_params = {}
    env = LunarHazardEnvironment(**env_params)
    env = Monitor(env)

    base_model_params = {
        "gamma": 0.999,
        "gae_lambda": 0.99,
    }

    if model_params is None:
        model_params = {}
    base_model_params |= model_params

    model = PPO(policy="MlpPolicy", env=env, verbose=0, seed=1234, **base_model_params)

    eval_callback = TrainingEvaluationCallback()
    if record_clipping:
        diagnostics_callback = PPOTrainingDiagnosticsCallback()
        callbacks = [eval_callback, diagnostics_callback]
    else:
        callbacks = eval_callback

    print(f"Training model {filename}...")
    model.learn(
        total_timesteps=100_000,
        callback=callbacks,
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
        std_speeds=eval_callback.std_speeds,
        mean_target_errors=eval_callback.mean_target_errors,
        std_target_errors=eval_callback.std_target_errors,
        mean_control_efforts=eval_callback.mean_control_efforts,
        std_control_efforts=eval_callback.std_control_efforts,
        min_safeties=eval_callback.min_safeties,
    )

    if record_clipping:
        np.savez(
            f"ppo_diagnostics_{filename}.npz",
            steps=diagnostics_callback.steps,
            clip_fraction=diagnostics_callback.clip_fractions,
        )


if __name__ == "__main__":
    train_with_parameters("base_model")

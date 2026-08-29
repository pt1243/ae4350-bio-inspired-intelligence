import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hazard_maps import generate_hazard_map, get_local_patch


class LunarHazardEnvironment(gym.Env):
    def __init__(self):
        super().__init__()

        # simulation/dynamics parameters
        self.dt = 1.0  # s
        self.descent_time = 120  # s
        self.max_acceleration = 0.1  # m/s^2
        self.map_half_width = 50  # m, overall hazard map size
        self.map_size = self.map_half_width * 2
        self.patch_size = 7  # grid size of hazard map sample

        # action and observation setup
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)  # x and y acceleration outputs
        obs_size = (
            2  # predicted touchdown position
            + 2  # velocity
            + 1  # time remaining
            + self.patch_size**2  # hazard patch
        )
        # unlikely to encounter anything greater than 1000
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(obs_size,), dtype=np.float32)

        # reward parameters
        self.control_weight = 0.01  # propellant expenditure during descent, measured by acceleration actions
        self.velocity_weight = 10  # horizontal touchdown velocity
        self.safety_weight = 100  # hazard level at touchdown
        self.target_weight = 0.2  # weighting for deviating from selected touchdown spot at [0, 0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.hazard_map = generate_hazard_map(
            size=self.map_size, random_seed=int(self.np_random.integers(0, np.iinfo(np.int32).max))
        )[0]

        # TODO: add a downrange initial velocity and position component
        self.position = np.array([0, 0], dtype=float)
        self.velocity = self.np_random.uniform(-0.5, 0.5, size=2)

        self.time_remaining = self.descent_time

        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # compute acceleration, and update dynamics using simple forward Euler
        action = np.clip(action, -1.0, 1.0)
        acceleration = action * self.max_acceleration
        self.position += self.velocity * self.dt
        self.velocity += acceleration * self.dt

        self.time_remaining = max(0, self.time_remaining - self.dt)

        reward = self._get_running_reward(acceleration)

        info = {}
        terminated = self.time_remaining <= 0
        if terminated:
            reward += self._get_terminal_reward()
            info = self._get_info()

        observation = self._get_observation()

        return observation, reward, terminated, False, info

    def _get_observation(self):
        # returns [position error, velocity, time left, local hazard map patch]
        #
        # the targeted landing site is situated at (0, 0) every time, so target error can be measured by just the
        # position (instead of needing a distance)
        touchdown_prediction = self.position + self.velocity * self.time_remaining

        scaled_position = touchdown_prediction / self.map_half_width
        local_patch = get_local_patch(self.hazard_map, scaled_position[0], scaled_position[1], self.patch_size)

        scaled_velocity = self.velocity / 5.0  # TODO: update
        scaled_time = self.time_remaining / self.descent_time

        observation = np.concatenate([scaled_position, scaled_velocity, [scaled_time], local_patch.flatten()])

        return observation.astype(np.float32)

    def _get_touchdown_safety(self) -> float:
        normalized_position = self.position / self.map_half_width

        if not np.all((normalized_position >= -1.0) & (normalized_position <= 1.0)):
            return 0.0  # outside the map is treated as fully unsafe

        ix = int(np.round((normalized_position[0] + 1.0) / 2.0 * (self.map_size - 1)))
        iy = int(np.round((normalized_position[1] + 1.0) / 2.0 * (self.map_size - 1)))
        return float(self.hazard_map[iy, ix])

    def _get_running_reward(self, acceleration) -> float:
        # apply a running penalty to high control inputs as a proxy for propellant consumption
        return -self.control_weight * np.sum(acceleration**2) * self.dt

    def _get_terminal_reward(self) -> float:
        safety = self._get_touchdown_safety()

        ref_velocity = 1.0  # m/s  TODO: update?
        velocity_penalty = np.sum(self.velocity**2) / ref_velocity**2

        target_penalty = np.sum(self.position**2) / self.map_half_width**2

        return (
            self.safety_weight * safety - self.velocity_weight * velocity_penalty - self.target_weight * target_penalty
        )

    def _get_info(self) -> dict:
        return {
            "touchdown_safety": self._get_touchdown_safety(),
            "touchdown_speed": float(np.linalg.norm(self.velocity)),
            "target_error": float(np.linalg.norm(self.position)),
        }

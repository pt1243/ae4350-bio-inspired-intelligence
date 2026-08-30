import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hazard_maps import generate_hazard_map, get_local_patch


class LunarHazardEnvironment(gym.Env):
    def __init__(
        self,
        dt: float = 1.0,
        descent_time: float = 60,
        max_acceleration: float = 0.1,
        map_half_width: int = 50,
        patch_size: int = 7,
        nominal_downrange_velocity: float = 0.5,
        control_weight: float = 5,
        velocity_weight: float = 20,
        safety_weight: float = 100,
        target_weight: float = 20,
    ):
        super().__init__()

        # simulation/dynamics parameters
        self.dt = dt  # s
        self.descent_time = descent_time  # s
        self.max_acceleration = max_acceleration  # m/s^2
        self.map_half_width = map_half_width  # m, overall hazard map size
        self.map_size = self.map_half_width * 2
        self.patch_size = patch_size  # grid size of hazard map sample
        self.nominal_downrange_velocity = nominal_downrange_velocity  # m/s

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
        self.control_weight = control_weight  # propellant expenditure during descent, measured by acceleration actions
        self.velocity_weight = velocity_weight  # horizontal touchdown velocity
        self.safety_weight = safety_weight  # hazard level at touchdown
        self.target_weight = target_weight  # weighting for deviating from selected touchdown spot at [0, 0]

    def reset(self, seed=None):
        super().reset(seed=seed)

        self.hazard_map = generate_hazard_map(
            size=self.map_size, random_seed=int(self.np_random.integers(0, np.iinfo(np.int32).max))
        )[0]

        self.position = np.array([-self.nominal_downrange_velocity * self.descent_time, 0], dtype=float)
        self.velocity = np.array(
            [
                self.np_random.uniform(self.nominal_downrange_velocity * 0.5, self.nominal_downrange_velocity * 1.5),
                self.np_random.uniform(-0.25, 0.25),
            ]
        )

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

        scaled_velocity = self.velocity
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

        velocity_penalty = np.sum(self.velocity**2)

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

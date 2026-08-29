import numpy as np
from gymnasium.utils.env_checker import check_env

from environment import LunarHazardEnvironment


def run_constant_action(action):
    env = LunarHazardEnvironment()

    obs, info = env.reset(seed=42)

    initial_position = env.position.copy()
    initial_velocity = env.velocity.copy()

    terminated = False
    truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))

        total_reward += reward

    return {
        "initial_position": initial_position,
        "initial_velocity": initial_velocity,
        "final_position": env.position.copy(),
        "final_velocity": env.velocity.copy(),
        "total_reward": total_reward,
    }


def expected_final_state(
    initial_position,
    initial_velocity,
    action,
    max_acceleration,
    descent_time,
):
    action = np.asarray(action, dtype=float)

    acceleration = action * max_acceleration

    expected_position = initial_position + initial_velocity * descent_time + 0.5 * acceleration * descent_time**2

    expected_velocity = initial_velocity + acceleration * descent_time

    return expected_position, expected_velocity


if __name__ == "__main__":
    env = LunarHazardEnvironment()
    check_env(env)

    test_actions = {
        "zero acceleration": [0.0, 0.0],
        "positive x": [1.0, 0.0],
        "negative x": [-1.0, 0.0],
    }

    for name, action in test_actions.items():
        result = run_constant_action(action)

        print(f"\n{name}")
        print("-" * 40)
        print("Initial position:", result["initial_position"])
        print("Initial velocity:", result["initial_velocity"])
        print("Final position:  ", result["final_position"])
        print("Final velocity:  ", result["final_velocity"])
        print("Total reward:    ", result["total_reward"])

        expected_position, expected_velocity = expected_final_state(
            result["initial_position"],
            result["initial_velocity"],
            action,
            max_acceleration=0.1,
            descent_time=120.0,
        )

        print("Expected position:", expected_position)
        print("Actual position:  ", result["final_position"])

        print("Expected velocity:", expected_velocity)
        print("Actual velocity:  ", result["final_velocity"])

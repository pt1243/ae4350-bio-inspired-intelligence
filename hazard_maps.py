import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom


def generate_hazard_map(size: int = 100, coarse_scale: int = 8, coarse_weight: float = 0.4, n_hazards: int = 12, random_seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    # for different resolutions, use [-1, 1] as the boundaries. convert back to meters later
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # large scale background noise, meant to represent general terrain characteristics for this area
    # (e.g. lowlands, hills, smooth maria, etc)
    # approximate something like Perlin noise, which varies smoothly over a large scale
    coarse_size = size // coarse_scale
    coarse_noise = zoom(
        rng.random((coarse_size, coarse_size)),
        zoom=size / coarse_size,
        order=3
    )
    # ensure that interpolation returned the correct size, and normalize
    coarse_noise = coarse_noise[:size, :size]
    coarse_noise -= np.min(coarse_noise)
    coarse_noise /= np.max(coarse_noise) + 1e-8

    coarse_cost = coarse_noise * coarse_weight


    crater_cost = np.zeros_like(coarse_cost)
    # small scale crater hazards: Gaussian blobs
    for _ in range(n_hazards):
        crater_x = rng.uniform(-1, 1)
        crater_y = rng.uniform(-1, 1)
        sigma = rng.uniform(0.03, 0.25)  # assume roughly circular
        severity = rng.uniform(0.5, 1)  # some craters are worse than others

        gaussian_cost = severity * np.exp(
            -0.5 * (((X - crater_x) / sigma)**2 + ((Y - crater_y) / sigma)**2)
        )
        crater_cost += gaussian_cost

    # convert from a hazard cost to a hazard map
    # high cost is associated with low safety
    hazard_cost = coarse_cost + crater_cost
    hazard_map = np.exp(-hazard_cost)
    hazard_map -= np.min(hazard_map)
    hazard_map /= np.max(hazard_map) + 1e-8

    # return intermediate values to show a side-by-side figure in the report
    return hazard_map, hazard_cost, coarse_cost, crater_cost



def show_hazard_map(hazard_map: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    size = hazard_map.shape[0] / 2
    img = ax.imshow(hazard_map, extent=[-size, size, -size, size])
    plt.colorbar(img, ax=ax, label="Landing safety")

    ax.set_title("Hazard map")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.tight_layout()

if __name__ == "__main__":
    for seed in range(10):
        show_hazard_map(generate_hazard_map(coarse_weight=0.4, random_seed=seed)[0])

    plt.show()
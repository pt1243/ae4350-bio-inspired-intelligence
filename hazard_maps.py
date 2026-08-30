import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom


def generate_hazard_map(
    size: int = 100, coarse_scale: int = 8, coarse_weight: float = 0.4, n_hazards: int = 12, random_seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)

    # note: these are normalized coordinates; convert back to meters later
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # large scale background noise, meant to represent general terrain characteristics for this area
    # (e.g. lowlands, hills, smooth maria, etc)
    # approximate something like Perlin noise, which varies smoothly over a large scale
    coarse_size = size // coarse_scale
    coarse_noise = zoom(rng.random((coarse_size, coarse_size)), zoom=size / coarse_size, order=3)
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

        gaussian_cost = severity * np.exp(-0.5 * (((X - crater_x) / sigma) ** 2 + ((Y - crater_y) / sigma) ** 2))
        crater_cost += gaussian_cost

    # convert from a hazard cost to a hazard map
    # high cost is associated with low safety
    hazard_cost = coarse_cost + crater_cost
    hazard_map = np.exp(-hazard_cost)
    hazard_map -= np.min(hazard_map)
    hazard_map /= np.max(hazard_map) + 1e-8

    # return intermediate values to show a side-by-side figure in the report
    return hazard_map, hazard_cost, coarse_cost, crater_cost


def get_local_patch(hazard_map: np.ndarray, x: float, y: float, patch_size: int) -> np.ndarray:
    if patch_size % 2 == 0:
        raise ValueError("patch size must be an odd number")

    size = hazard_map.shape[0]
    half_patch_size = patch_size // 2

    # convert to pixel indices
    ix = int(np.round((x + 1.0) / 2.0 * (size - 1)))
    iy = int(np.round((y + 1.0) / 2.0 * (size - 1)))

    # pad with zeros, in case we are near the edge of the map
    padded_map = np.pad(hazard_map, pad_width=half_patch_size, mode="constant", constant_values=0.0)

    # adjust coordinates to account for padding
    ix += half_patch_size
    iy += half_patch_size

    x0 = ix - half_patch_size
    x1 = ix + half_patch_size + 1
    y0 = iy - half_patch_size
    y1 = iy + half_patch_size + 1
    if x0 >= 0 and x1 <= padded_map.shape[1] and y0 >= 0 and y1 <= padded_map.shape[0]:
        patch = padded_map[y0:y1, x0:x1]
    else:  # fully outside
        patch = np.zeros((patch_size, patch_size), dtype=hazard_map.dtype)
    return patch


def plot_hazard_map(hazard_map: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))

    size = hazard_map.shape[0] / 2
    img = ax.imshow(hazard_map, extent=[-size, size, -size, size])
    plt.colorbar(img, ax=ax, label="Landing safety")

    ax.set_title("Hazard map")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    fig.tight_layout()


def plot_hazard_map_components(hazard_map: np.ndarray, coarse_cost: np.ndarray, crater_cost: np.ndarray):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    size = hazard_map.shape[0] / 2

    hazard_cmap = "OrRd"
    safety_cmap = "viridis"
    hazard_min = min(np.min(coarse_cost), np.min(crater_cost))
    hazard_max = max(np.max(coarse_cost), np.max(crater_cost))

    img_0 = axs[0].imshow(
        coarse_cost,
        extent=[-size, size, -size, size],
        origin="lower",
        cmap=hazard_cmap,
        vmin=hazard_min,
        vmax=hazard_max,
    )
    fig.colorbar(img_0, ax=axs[0], label="Hazard level", fraction=0.045, pad=0.04)
    axs[0].set_title("Large-scale terrain")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")

    img_1 = axs[1].imshow(
        crater_cost,
        extent=[-size, size, -size, size],
        origin="lower",
        cmap=hazard_cmap,
        vmin=hazard_min,
        vmax=hazard_max,
    )
    fig.colorbar(img_1, ax=axs[1], label="Hazard level", fraction=0.045, pad=0.04)
    axs[1].set_title("Crater hazards")
    axs[1].set_xlabel("x [m]")
    axs[1].set_ylabel("y [m]")

    img_2 = axs[2].imshow(
        hazard_map, extent=[-size, size, -size, size], origin="lower", cmap=safety_cmap, vmin=0, vmax=1
    )
    fig.colorbar(img_2, ax=axs[2], label="Safety", fraction=0.045, pad=0.04)
    axs[2].set_title("Combined hazard map")
    axs[2].set_xlabel("x [m]")
    axs[2].set_ylabel("y [m]")

    for ax in axs:
        ax.set_aspect("equal")

    fig.tight_layout()

    fig.savefig("plots/hazard_maps.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    # for seed in range(10):
    #     plot_hazard_map(generate_hazard_map(coarse_weight=0.4, random_seed=seed)[0])
    hazard_map, _, coarse, craters = generate_hazard_map(random_seed=0)
    plot_hazard_map_components(hazard_map, coarse, craters)
    # plt.show()

from dataclasses import dataclass

import numpy as np


@dataclass
class LBMDomain:
    points_coordinates: tuple[np.ndarray, np.ndarray, np.ndarray]
    size = list[int]
    n_dim: int

    def __init__(self, domain_sizes: list[int]):
        self.n_dim = len(domain_sizes)
        self.size = domain_sizes
        if len(domain_sizes) == 2:
            self.size.append(1)
        x, y, z = (np.arange(0, domain_sizes[i], 1) for i in range(3))
        self.points_coordinates = np.meshgrid(
            x, y, z, indexing="ij"  # X then Y then Z
        )[::-1]

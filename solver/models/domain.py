from dataclasses import dataclass

import numpy as np


@dataclass
class DomainBox:
    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int

    # TODO: Validate class so that *_max >= *_min


class LBMDomain:
    domain_box: DomainBox

    def __init__(
        self, x_min: int, x_max: int, y_min: int, y_max: int, z_min: int, z_max: int
    ):
        self.domain_box = DomainBox(x_min, x_max, y_min, y_max, z_min, z_max)
        self.domain_x = np.arange(x_min, x_max, 1)
        self.domain_y = np.arange(y_min, y_max, 1)
        self.domain_z = np.arange(z_min, z_max, 1)
        self.domain_nodes = np.meshgrid(
            self.domain_x,
            self.domain_y,
            self.domain_z,
            indexing="ij",  # X then Y then Z
        )[::-1]

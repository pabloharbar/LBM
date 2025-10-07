from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationScalarField:
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    uz: np.ndarray
    f: np.ndarray

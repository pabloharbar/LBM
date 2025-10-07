from dataclasses import dataclass
from enum import Enum

import numpy as np


class VelocitySetType(Enum):
    D2Q9 = "D2Q9"
    D3Q19 = "D3Q19"
    D3Q27 = "D3Q27"


@dataclass(frozen=True)
class VelocitySet:
    set_type: VelocitySetType
    order: int
    directions: np.ndarray  # Shape: (Q, D)
    weights: np.ndarray  # Shape: (Q,)
    cs: float


class VelocitySetFactory:
    _sets = {
        VelocitySetType.D2Q9: {
            "cs": 1 / np.sqrt(3),
            "order": 2,
            "directions": np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [-1, 0, 0],
                    [0, -1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                ]
            ),
            "weights": np.array(
                [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36]
            ),
        },
        VelocitySetType.D3Q19: {
            "cs": 1 / np.sqrt(3),
            "order": 2,
            "directions": np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [1, 0, 1],
                    [-1, 0, 1],
                    [1, 0, -1],
                    [-1, 0, -1],
                    [0, 1, 1],
                    [0, -1, 1],
                    [0, 1, -1],
                    [0, -1, -1],
                ]
            ),
            "weights": np.array(
                [
                    1 / 3,
                    1 / 18,
                    1 / 18,
                    1 / 18,
                    1 / 18,
                    1 / 18,
                    1 / 18,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                    1 / 36,
                ]
            ),
        },
        VelocitySetType.D3Q27: {
            "cs": 1 / np.sqrt(3),
            "order": 3,
            "directions": np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, -1, 0],
                    [0, 0, 1],
                    [0, 0, -1],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [1, -1, 0],
                    [-1, -1, 0],
                    [1, 0, 1],
                    [-1, 0, 1],
                    [1, 0, -1],
                    [-1, 0, -1],
                    [0, 1, 1],
                    [0, -1, 1],
                    [0, 1, -1],
                    [0, -1, -1],
                    [1, 1, 1],
                    [-1, 1, 1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [1, -1, -1],
                    [-1, -1, -1],
                ]
            ),
            "weights": np.array(
                [
                    8 / 27,
                    2 / 27,
                    2 / 27,
                    2 / 27,
                    2 / 27,
                    2 / 27,
                    2 / 27,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 54,
                    1 / 216,
                    1 / 216,
                    1 / 216,
                    1 / 216,
                    1 / 216,
                    1 / 216,
                    1 / 216,
                    1 / 216,
                ]
            ),
        },
    }

    @staticmethod
    def create(set_name: str) -> VelocitySet:
        available_sets = [k.value for k in VelocitySetFactory._sets.keys()]
        if set_name not in available_sets:
            raise ValueError(
                f"Velocity set '{set_name}' is not implemented, availables are {available_sets}."
            )
        set_type = VelocitySetType(set_name)
        config = VelocitySetFactory._sets[set_type]

        return VelocitySet(
            set_type=set_type,
            cs=config["cs"],
            directions=config["directions"],
            weights=config["weights"],
        )

from enum import Enum


class VelocitySetTypes(Enum):
    D2Q9 = "D2Q9"


class BaseVelocitySet:
    vel_set: VelocitySetTypes

    def get_vel_set(self):
        raise NotImplementedError()


class VelocitySetD2Q9:
    vel_set = VelocitySetTypes.D2Q9

    def get_vel_set(self):
        Nx = 1
        return {
            "e0": {"direction": [0, 0, 0], "weight": 16 / 36, "index_offset": 0},
            "e1": {"direction": [1, 0, 0], "weight": 4 / 36, "index_offset": 1},
            "e2": {"direction": [0, 1, 0], "weight": 4 / 36, "index_offset": Nx},
            "e3": {"direction": [-1, 0, 0], "weight": 4 / 36, "index_offset": -1},
            "e4": {"direction": [0, -1, 0], "weight": 4 / 36, "index_offset": -Nx},
            "e5": {"direction": [1, 1, 0], "weight": 1 / 36, "index_offset": Nx + 1},
            "e6": {"direction": [-1, 1, 0], "weight": 1 / 36, "index_offset": Nx - 1},
            "e7": {"direction": [-1, -1, 0], "weight": 1 / 36, "index_offset": -Nx - 1},
            "e8": {"direction": [1, -1, 0], "weight": 1 / 36, "index_offset": -Nx + 1},
        }

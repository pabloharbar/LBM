from enum import Enum


class VelocitySetTypes(Enum):
    D2Q9 = "D2Q9"


class BaseVelocitySet:
    set_type: VelocitySetTypes

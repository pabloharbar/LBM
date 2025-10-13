from __future__ import annotations

import pathlib
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from solver.utils import read_yaml


class BCTypes(str, Enum):
    bounce_back = "Bounce-Back"
    zou_he = "Zou-He"


class DomainFaces(str, Enum):
    x_plus = "x_plus"
    x_minus = "x_minus"
    y_plus = "y_plus"
    y_minus = "y_minus"
    z_plus = "z_plus"
    z_minus = "z_minus"

    @property
    def inside_normal_value(self):
        if not hasattr(self, "_normal"):
            self._normal = 1 if "minus" in self.value else -1
        return self._normal

    @property
    def normal_axis(self):
        if not hasattr(self, "_n_axis"):
            self._n_axis = ["x", "y", "z"].index(self.value.split("_")[0])
        return self._n_axis

    def get_target_index(self, domain_sizes: list[int]):
        if self.inside_normal_value == -1:
            return 0
        else:
            return domain_sizes[self.normal_axis] - 1


class BoundaryConditionConfig(BaseModel):
    bc_type: BCTypes
    u: float = Field(None, description="Boundary velocity 'u' component")
    v: float = Field(None, description="Boundary velocity 'v' component")
    w: float = Field(None, description="Boundary velocity 'w' component")
    rho: float = Field(None, description="Boundary density 'rho' component")

    @model_validator(mode="after")
    def validate_bc_type(self) -> BoundaryConditionConfig:
        if self.bc_type == BCTypes.bounce_back and any(
            [self.u, self.v, self.w, self.rho]
        ):
            raise ValueError(
                f"When using '{self.bc_type.value}' type, do not set u, v, w nor rho"
            )
        return self


class LBMSimulationConfig(BaseModel):
    sim_name: str = Field(..., title="Name of the simulation")
    domain_size: list[int] = Field(..., title="Domain size as [Nx, Ny, Nz] or [Nx, Ny]")
    n_timesteps: int = Field(..., title="Number of timesteps", gt=0)
    export_frequency: int = Field(
        0, title="Number of timesteps intervals to export", ge=0
    )
    velocity_set: str = Field(..., title="Velocity set to use")
    tau: float = Field(..., title="Tau value for BGK operator", gt=0.5)
    boundary_conditions: dict[DomainFaces, BoundaryConditionConfig] = Field(
        ..., description="Boundary conditions definition"
    )

    @model_validator(mode="after")
    def validate_bcs(self) -> LBMSimulationConfig:
        n_dim = len(self.domain_size)
        expected_bc_keys = sorted(
            [f.value for f in DomainFaces if "z" not in f.value or n_dim == 3]
        )
        if sorted([k for k in self.boundary_conditions.keys()]) != expected_bc_keys:
            raise ValueError(f"Expected boundary conditions keys {expected_bc_keys}")
        return self

    @field_validator("domain_size", mode="after")
    def validate_domain_size(cls, vals):
        if len(vals) not in (2, 3) or any(v <= 0 for v in vals):
            raise Exception(
                "Domain size must have 2 or 3 dimensions and all must be positive integers"
            )
        return vals

    @classmethod
    def from_file(cls, file_path: pathlib.Path):
        if file_path.exists():
            yaml_vals = read_yaml(file_path)
            params = cls(**yaml_vals)
            return params
        else:
            raise Exception(
                f"Unable to read yaml. Filepath {file_path} does not exists"
            )

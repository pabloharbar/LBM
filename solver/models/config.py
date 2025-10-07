from __future__ import annotations

import pathlib
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from solver.utils import read_yaml


class BCTypes(str, Enum):
    bounce_back = "Bounce-Back"
    zhou_he = "Zhou-He"


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
    domain_size: list[int] = Field(..., title="Domain size as [Nx, Ny, Nz] or [Nx, Ny]")
    n_timesteps: int = Field(..., title="Number of timesteps", gt=0)
    export_frequency: int = Field(
        0, title="Number of timesteps intervals to export", ge=0
    )
    velocity_set: str = Field(..., title="Velocity set to use")
    tau: float = Field(..., title="Tau value for BGK operator", gt=0.5)
    boundary_conditions: dict[str, BoundaryConditionConfig] = Field(
        ..., description="Boundary conditions definition"
    )

    @model_validator(mode="after")
    def validate_bcs(self) -> LBMSimulationConfig:
        n_dim = len(self.domain_size)
        expected_bc_keys = ["x_plus", "x_minus", "y_plus", "y_minus"]
        if n_dim == 3:
            expected_bc_keys += ["z_plus", "z_minus"]
        if sorted([k for k in self.boundary_conditions.keys()]) != sorted(
            expected_bc_keys
        ):
            raise ValueError(f"Expected boundary conditions keys {expected_bc_keys}")
        return self

    @field_validator("domain_size", mode="after")
    def validate_domain_size(cls, vals):
        if len(vals) not in (2, 3) or any(v <= 0 for v in vals):
            raise Exception(
                "Domain size must have 2 or 3 dimensions and all must be positive integers"
            )
        if len(vals) < 3:
            vals.append(1)
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

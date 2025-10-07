import pathlib

import numpy as np
import pyvista as pv

from solver.models.domain import LBMDomain
from solver.simulation.data import SimulationData


def export_macros(
    timestep: int,
    data: SimulationData,
    domain: LBMDomain,
):
    grid = pv.StructuredGrid(*domain.points_coordinates)
    grid.point_data["rho"] = data.rho.ravel()
    grid.point_data["ux"] = data.ux.ravel()
    grid.point_data["uy"] = data.uy.ravel()
    grid.point_data["uz"] = data.uz.ravel()
    output_filename = pathlib.Path("./cases") / f"macros_{timestep:07d}.vts"
    grid.save(output_filename, binary=True)


def export_population(timestep: int, data: SimulationData, domain: LBMDomain):
    grid = pv.StructuredGrid(*domain.points_coordinates)
    for i in range(data.f.shape[2]):
        grid.point_data[f"f{i}"] = data.f[:, :, i].ravel()
    output_filename = pathlib.Path("./cases") / f"population_{timestep:07d}.vts"
    grid.save(output_filename, binary=True)

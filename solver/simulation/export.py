import pathlib

import numpy as np
import pyvista as pv

from solver.models.domain import LBMDomain
from solver.simulation.data import SimulationData


def export_macros(
    data: SimulationData,
    domain: LBMDomain,
    target_path: pathlib.Path,
):
    grid = pv.StructuredGrid(
        domain.points_coordinates[0],
        domain.points_coordinates[1],
        domain.points_coordinates[2],
    )
    grid.point_data["rho"] = data.rho.ravel()
    grid.point_data["ux"] = data.ux.ravel()
    grid.point_data["uy"] = data.uy.ravel()
    grid.point_data["uz"] = data.uz.ravel()
    grid.save(target_path, binary=True)


def export_population(
    data: SimulationData, domain: LBMDomain, target_path: pathlib.Path
):
    grid = pv.StructuredGrid(*domain.points_coordinates)
    for i in range(data.f.shape[2]):
        grid.point_data[f"f{i}"] = data.f[:, :, :, i].ravel()
    grid.save(target_path, binary=True)

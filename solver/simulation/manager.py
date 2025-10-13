import pathlib

import numpy as np

from solver.models.config import BCTypes, LBMSimulationConfig
from solver.models.domain import LBMDomain
from solver.models.velocity_set import VelocitySetFactory
from solver.simulation.boundary_conditions import bounce_back, zou_he
from solver.simulation.data import SimulationData
from solver.simulation.equations import equilibrium
from solver.simulation.export import export_macros, export_population


class SimulationManager:
    def __init__(self, cfg: LBMSimulationConfig):
        self.cfg = cfg
        self.results_path = pathlib.Path("./results") / self.cfg.sim_name

    def initialize_simulation(self):
        self.domain = LBMDomain(self.cfg.domain_size)
        rho0 = np.ones(self.domain.size)
        u0 = np.zeros(self.domain.size)
        self.vel_set = VelocitySetFactory.create(self.cfg.velocity_set)
        self.sim_data = SimulationData(
            rho=rho0, ux=u0, uy=u0, uz=u0, f=equilibrium(rho0, u0, u0, u0, self.vel_set)
        )
        export_macros(
            self.sim_data, self.domain, self.results_path / f"macros_{0:07d}.vts"
        )
        export_population(self.sim_data, self.domain / f"population_{0:07d}.vts")

    def apply_bcs(self):
        for bc_face, bc_cfg in self.cfg.boundary_conditions.items():
            if bc_cfg.bc_type == BCTypes.bounce_back:
                bounce_back(
                    self.cfg.velocity_set, bc_face, self.domain.size, self.sim_data.f
                )
            elif bc_cfg.bc_type == BCTypes.zou_he:
                zou_he(self.cfg.velocity_set, bc_face)

    def run_simulation(self):
        e = self.vel_set.directions
        for it in range(1, self.cfg.n_timesteps + 1):
            # --- Streaming Step ---
            for i in range(len(e)):
                self.sim_data.f[:, :, :, i] = np.roll(
                    self.sim_data.f[:, :, :, i], e[i, :], axis=(0, 1, 2)
                )
            # --- Boundary Conditions ---
            self.apply_bcs()

            # --- Macroscopic Variables ---
            f = self.sim_data.f
            rho = np.sum(f, axis=3)
            ux = (np.sum(f * e[:, 0], axis=3)) / rho
            uy = (np.sum(f * e[:, 1], axis=3)) / rho
            uz = (np.sum(f * e[:, 2], axis=3)) / rho

            # --- Collision Step ---
            f_eq = equilibrium(rho, ux, uy, uz, self.vel_set)
            f += (f_eq - f) / self.cfg.tau

            # --- Export and save data ---
            self.sim_data = SimulationData(rho, ux, uy, uz, f)
            if it % self.cfg.export_frequency == 0:
                export_macros(
                    self.sim_data,
                    self.domain,
                    self.results_path / f"macros_{it:07d}.vts",
                )
                export_population(
                    self.sim_data,
                    self.domain,
                    self.results_path / f"population_{it:07d}.vts",
                )
                print("Exported macros!")
            print(f"Iteration: {it}/{self.cfg.n_timesteps}")
        print("Simulation finished.")

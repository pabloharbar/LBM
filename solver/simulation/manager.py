import numpy as np

from solver.models.config import BCTypes, LBMSimulationConfig
from solver.models.domain import LBMDomain
from solver.models.velocity_set import VelocitySetFactory
from solver.simulation.data import SimulationData
from solver.simulation.equations import equilibrium
from solver.simulation.export import export_macros, export_population


class SimulationManager:
    def __init__(self, cfg: LBMSimulationConfig):
        self.cfg = cfg

    def initialize_simulation(self):
        self.domain = LBMDomain(self.cfg.domain_size)
        self.vel_set = VelocitySetFactory.create(self.cfg.velocity_set)
        self.sim_data = SimulationData(
            rho=np.ones(self.domain.size),
            ux=np.zeros(self.domain.size),
            uy=np.zeros(self.domain.size),
            uz=np.zeros(self.domain.size),
            f=equilibrium(self.rho, self.ux, self.uy, self.uz),
        )
        export_macros(0, self.sim_data, self.domain)
        export_population(0, self.sim_data, self.domain)
        bc_target_index = {
            "x_plus": {
                "idx_range": {
                    "x": [0, 1],
                    "y": [0, 1],
                    "z": [0, 1],
                },
                "bounce_back": {"from": [], "to": []},
            }
        }

    def run_simulation(self):
        e = self.vel_set.directions
        for it in range(1, self.cfg.n_timesteps + 1):
            # --- Streaming Step ---
            for i in range(len(e)):
                self.f[:, :, :, i] = np.roll(self.f[:, :, i], e[i, :], axis=(0, 1, 2))
                # self.f[:, :, :, i] = np.roll(
                #     np.roll(np.roll(self.f[:, :, i], e[i, 0], axis=0), e[i, 1], axis=1),
                #     e[i, 2],
                #     axis=2,
                # )

            # --- Boundary Conditions ---
            # 1. Bounce-back on solid walls
            # for bc_name, bc_cfg in self.cfg.boundary_conditions.items():
            #     if bc_cfg.bc_type == BCTypes.bounce_back:
            #         if "y" in bc_name:
            #             if "minus" in bc_name:
            #                 f[:, 0, [2, 5, 6]] = f[:, 0, [4, 7, 8]]
            #             else:

            # # Bottom wall (y=0)
            # f[:, 0, [2, 5, 6]] = f[:, 0, [4, 7, 8]]
            # # Left wall (x=0)
            # f[0, :, [1, 5, 8]] = f[0, :, [3, 7, 6]]
            # # Right wall (x=Nx-1)
            # f[Nx - 1, :, [3, 6, 7]] = f[Nx - 1, :, [1, 8, 5]]

            # # 2. Moving lid boundary condition (top wall) Zou He et. al
            # rho_lid = (
            #     f[:, Ny - 1, 0]
            #     + f[:, Ny - 1, 1]
            #     + f[:, Ny - 1, 3]
            #     + 2 * (f[:, Ny - 1, 2] + f[:, Ny - 1, 5] + f[:, Ny - 1, 6])
            # )
            # f[:, Ny - 1, 4] = f[:, Ny - 1, 2]
            # f[:, Ny - 1, 7] = (
            #     f[:, Ny - 1, 5]
            #     + (f[:, Ny - 1, 1] - f[:, Ny - 1, 3]) / 2
            #     - rho_lid * u_lid / 2
            # )
            # f[:, Ny - 1, 8] = (
            #     f[:, Ny - 1, 6]
            #     - (f[:, Ny - 1, 1] - f[:, Ny - 1, 3]) / 2
            #     + rho_lid * u_lid / 2
            # )

            # --- Macroscopic Variables ---
            # Calculate density and velocity from the distribution functions
            rho = np.sum(f, axis=2)
            ux = (np.sum(f * e[:, 0], axis=2)) / rho
            uy = (np.sum(f * e[:, 1], axis=2)) / rho
            uz = (np.sum(f * e[:, 2], axis=2)) / rho

            # --- Collision Step ---
            f_eq = equilibrium(rho, ux, uy)
            f += (f_eq - f) / self.cfg.tau

            self.sim_data = SimulationData(rho, ux, uy, uz, f)

            if it % self.cfg.export_frequency == 0:
                export_macros(it, self.sim_data, self.domain)
                export_population(it, self.sim_data, self.domain)
                print("Exported macros!")

            print(f"Iteration: {it}/{self.cfg.n_timesteps}")

        print("Simulation finished.")

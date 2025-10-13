import numpy as np

from solver.models.config import DomainFaces
from solver.models.velocity_set import VelocitySet


def bounce_back(
    velocity_set: VelocitySet,
    domain_face: DomainFaces,
    domain_sizes: list[int],
    f: np.ndarray,
):
    target_index = domain_face.get_target_index(domain_sizes)
    source_vel_set_idx, target_vel_set_idx = velocity_set.bounce_back_idx(
        domain_face.normal_axis, "minus" in domain_face.value
    )
    if "x" in domain_face.value:
        f[target_index, :, :, target_vel_set_idx] = f[
            target_index, :, :, source_vel_set_idx
        ]
    elif "y" in domain_face.value:
        f[:, target_index, :, target_vel_set_idx] = f[
            :, target_index, :, source_vel_set_idx
        ]
    elif "z" in domain_face.value:
        f[:, :, target_index, target_vel_set_idx] = f[
            :, :, target_index, source_vel_set_idx
        ]


def zou_he(
    velocity_set: VelocitySet,
    domain_face: DomainFaces,
    domain_sizes: list[int],
    f: np.ndarray,
):
    return
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

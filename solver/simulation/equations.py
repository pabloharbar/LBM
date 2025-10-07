import numpy as np

from solver.models.velocity_set import VelocitySet


def equilibrium(
    rho: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    velocity_set: VelocitySet,
):
    eq_pop = np.zeros(
        (ux.shape[0], ux.shape[1], ux.shape[2], len(velocity_set.directions)),
        dtype=np.float32,
    )
    e = velocity_set.directions
    w = velocity_set.weights

    H_0 = 1
    H_1 = (
        e[:, 0] * ux[:, :, np.newaxis]
        + e[:, 1] * uy[:, :, np.newaxis]
        + e[:, 2] * uz[:, :, np.newaxis]
    ) / velocity_set.cs**2
    H_2 = (
        ux[:, :, np.newaxis] ** 2 * (e[:, 0] ** 2 - 1)
        + ux[:, :, np.newaxis] * uy[:, :, np.newaxis] * (e[:, 0] * e[:, 1])
        + uy[:, :, np.newaxis] ** 2 * (e[:, 1] ** 2 - 1)
    ) / (2 * velocity_set.cs**4)
    H_3 = 0
    # if velocity_set.order > 2:
    #     (e[:, 0] * ux[:, :, np.newaxis] + e[:, 1] * uy[:, :, np.newaxis]) ** 3
    #     H_3 = (
    #         ux[:, :, np.newaxis] ** 2 * (e[:, 0] ** 2 - 1)
    #         + ux[:, :, np.newaxis] * uy[:, :, np.newaxis] * (e[:, 0] * e[:, 1])
    #         + uy[:, :, np.newaxis] ** 2 * (e[:, 1] ** 2 - 1)
    #     ) / (6 * velocity_set.cs**6)

    eq_pop = w * rho[:, :, np.newaxis] * (H_0 + H_1 + H_2 + H_3)
    return eq_pop

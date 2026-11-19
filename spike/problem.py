"""Transient heat conduction with a closed-form answer: mosaic#375 idea 4.

Slab [0, Lx] x [0, Ly], conductivity k_e = k0 * rho_e per cell (rho is the mosaic thermal-mesh field), unit heat
capacity. Cold left face (T = 0 at x = 0), all other faces insulated. With uniform rho = 1 the initial temperature
T0 = sin(pi x / (2 Lx)) is a single eigenmode and decays as

    T(x, t) = sin(pi x / (2 Lx)) exp(-lam t),    lam = k0 (pi / (2 Lx))^2.

Objective: J = integral of T(t_end)^2 over the slab = (Lx Ly / 2) exp(-2 lam t_end).
Its derivative along uniform conductivity (rho -> rho + eps * ones, i.e. k0 -> k0 (1 + eps)) is
dJ/deps = -2 lam t_end J. A backend's gradient with respect to rho, dotted with ones, must converge to that.

Discretisation shared by the backends: bilinear (Q1) finite elements on a structured nx x ny quad mesh, consistent
mass matrix, Dirichlet nodes at x = 0 eliminated, backward Euler with steps equal steps to t_end:

    (M + dt K(rho)) T_{n+1} = M T_n,    J_h = T_N' M T_N.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    nx: int = 32
    ny: int = 16
    Lx: float = 2.0
    Ly: float = 1.0
    k0: float = 1.0
    lam_t_end: float = 1.0  # t_end chosen so that lam * t_end is this; the exact J is then (Lx Ly / 2) e^-2

    @property
    def lam(self) -> float:
        return self.k0 * (np.pi / (2 * self.Lx)) ** 2

    @property
    def t_end(self) -> float:
        return self.lam_t_end / self.lam

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny

    def exact_J(self) -> float:
        return self.Lx * self.Ly / 2 * np.exp(-2 * self.lam * self.t_end)

    def exact_dJ_uniform(self) -> float:
        """dJ/deps for rho -> rho + eps * ones (k0 -> k0 (1 + eps))."""
        return -2 * self.lam * self.t_end * self.exact_J()


def element_matrices(hx: float, hy: float) -> tuple[np.ndarray, np.ndarray]:
    """Q1 stiffness (unit conductivity) and consistent mass on an hx x hy rectangle.

    Local node order: (0,0), (1,0), (1,1), (0,1), counter-clockwise.
    """
    kx = np.array([[2, -2, -1, 1], [-2, 2, 1, -1], [-1, 1, 2, -2], [1, -1, -2, 2]], dtype=float)
    ky = np.array([[2, 1, -1, -2], [1, 2, -2, -1], [-1, -2, 2, 1], [-2, -1, 1, 2]], dtype=float)
    ke = hy / (6 * hx) * kx + hx / (6 * hy) * ky
    me = hx * hy / 36 * np.array([[4, 2, 1, 2], [2, 4, 2, 1], [1, 2, 4, 2], [2, 1, 2, 4]], dtype=float)
    return ke, me


def mesh(p: Problem):
    """Node coordinates (n_nodes, 2), cell connectivity (n_cells, 4), indices of the free (non-Dirichlet) nodes."""
    xs = np.linspace(0.0, p.Lx, p.nx + 1)
    ys = np.linspace(0.0, p.Ly, p.ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="xy")  # node id = iy * (nx + 1) + ix
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)

    def nid(ix, iy):
        return iy * (p.nx + 1) + ix

    cells = np.array(
        [[nid(ix, iy), nid(ix + 1, iy), nid(ix + 1, iy + 1), nid(ix, iy + 1)] for iy in range(p.ny) for ix in range(p.nx)],
        dtype=np.int64,
    )
    free = np.flatnonzero(coords[:, 0] > 0.5 * p.Lx / p.nx)  # all nodes except x = 0
    return coords, cells, free


def initial_temperature(coords: np.ndarray, p: Problem) -> np.ndarray:
    return np.sin(np.pi * coords[:, 0] / (2 * p.Lx))


def assemble_dense(p: Problem, rho: np.ndarray):
    """Reference assembly (numpy, dense, free nodes only) for testing the backends: K(rho), M."""
    coords, cells, free = mesh(p)
    ke, me = element_matrices(p.Lx / p.nx, p.Ly / p.ny)
    n = len(coords)
    K = np.zeros((n, n))
    M = np.zeros((n, n))
    for e, c in enumerate(cells):
        K[np.ix_(c, c)] += p.k0 * rho[e] * ke
        M[np.ix_(c, c)] += me
    return K[np.ix_(free, free)], M[np.ix_(free, free)]


def discrete_exact(p: Problem, steps: int) -> tuple[float, float]:
    """J_h and dJ_h/deps of the discrete scheme itself, uniform rho = 1, in closed form.

    With K v_i = lam_i M v_i (M-orthonormal v_i) and T0 = sum c_i v_i, backward Euler gives
    T_N = sum c_i (1 + dt lam_i)^-N v_i, so J_h = sum c_i^2 (1 + dt lam_i)^-2N. Scaling the conductivity by
    (1 + eps) scales every lam_i, so dJ_h/deps = sum -2N c_i^2 dt lam_i (1 + dt lam_i)^(-2N-1).
    A backend's gradient compared with this shows its differentiation error alone, without the spatial and time
    discretisation error that the continuous closed form includes. Dense: practical up to a few thousand unknowns.
    """
    from scipy.linalg import eigh

    coords, _, free = mesh(p)
    K, M = assemble_dense(p, np.ones(p.n_cells))
    lam, V = eigh(K, M)  # V' M V = I
    c = V.T @ (M @ initial_temperature(coords, p)[free])
    dt = p.t_end / steps
    q = 1.0 + dt * lam
    J = float(np.sum(c**2 * q ** (-2.0 * steps)))
    dJ = float(np.sum(-2.0 * steps * c**2 * dt * lam * q ** (-2.0 * steps - 1)))
    return J, dJ


def reference_solve(p: Problem, rho: np.ndarray, steps: int) -> float:
    """Dense numpy backward Euler, for checking the backends on small meshes."""
    coords, _, free = mesh(p)
    K, M = assemble_dense(p, rho)
    dt = p.t_end / steps
    A = M + dt * K
    T = initial_temperature(coords, p)[free]
    lu = np.linalg.inv(A) @ M
    for _ in range(steps):
        T = lu @ T
    return float(T @ M @ T)


if __name__ == "__main__":
    ke, me = element_matrices(0.3, 0.2)
    assert np.allclose(ke.sum(axis=1), 0), "stiffness rows must sum to zero (constants are in the kernel)"
    assert np.isclose(me.sum(), 0.3 * 0.2), "mass must integrate to the element area"
    # a linear field x has energy integral |grad x|^2 = area
    x = np.array([0.0, 0.3, 0.3, 0.0])
    assert np.isclose(x @ ke @ x, 0.3 * 0.2)
    p = Problem(nx=16, ny=8)
    for steps in (10, 100, 1000):
        Jh = reference_solve(p, np.ones(p.n_cells), steps)
        print(f"steps {steps:5d}  J_h {Jh:.6f}  exact {p.exact_J():.6f}  rel err {abs(Jh / p.exact_J() - 1):.2e}")

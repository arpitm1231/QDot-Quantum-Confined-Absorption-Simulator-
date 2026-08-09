"""
Core quantum-dot physics engine.
Effective-mass approximation, 2D finite-difference Schrodinger equation,
solved with scipy's sparse Lanczos eigensolver (ARPACK).
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh

HBAR2_2ME = 3.81   # eV . Angstrom^2   (hbar^2 / 2*m_e)
HC = 1239.84       # eV . nm  (h*c)

MATERIALS = {
    "CdSe": {"label": "CdSe — Cadmium Selenide", "Eg": 1.74, "me": 0.13, "mh": 0.45},
    "PbS":  {"label": "PbS — Lead Sulfide",       "Eg": 0.41, "me": 0.085, "mh": 0.085},
    "InAs": {"label": "InAs — Indium Arsenide",   "Eg": 0.354, "me": 0.023, "mh": 0.41},
    "Si":   {"label": "Si — Silicon (reference)", "Eg": 1.12, "me": 0.26, "mh": 0.39},
    "InP":  {"label": "InP — Indium Phosphide (Cd-free)", "Eg": 1.35, "me": 0.077, "mh": 0.60},
    "CsPbBr3": {"label": "CsPbBr₃ — Perovskite QD (Cd-free)", "Eg": 2.3, "me": 0.20, "mh": 0.20},
    "FAPbI3":  {"label": "FAPbI₃ — Perovskite QD (Cd-free)",  "Eg": 1.48, "me": 0.20, "mh": 0.20},
}

SHAPES = ["circle", "square", "hexagon", "triangle"]


def inside_shape(shape, x, y, size_A):
    """x, y: arrays in Angstrom, relative to center. size_A: characteristic size."""
    r = size_A / 2
    if shape == "circle":
        return x**2 + y**2 <= r**2
    if shape == "square":
        return (np.abs(x) <= r) & (np.abs(y) <= r)
    if shape == "hexagon":
        inside = np.ones_like(x, dtype=bool)
        apo = r * np.cos(np.pi / 6)
        for k in range(6):
            th = np.pi / 3 * k
            inside &= (x * np.cos(th) + y * np.sin(th)) <= apo
        return inside
    if shape == "triangle":
        inside = np.ones_like(x, dtype=bool)
        apo = r * 0.5
        for k in range(3):
            th = np.pi / 2 + np.pi * 2 / 3 * k
            inside &= (x * np.cos(th) + y * np.sin(th)) <= apo
        return inside
    raise ValueError(f"unknown shape {shape}")


def build_grid(shape, size_nm, N, V0):
    """Returns (V as N x N array, dx in Angstrom)."""
    size_A = size_nm * 10
    domain_A = size_A * 2.4
    dx = domain_A / (N - 1)
    coords = np.arange(N) * dx - domain_A / 2
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    inside = inside_shape(shape, X, Y, size_A)
    V = np.where(inside, 0.0, V0)
    return V, dx


def build_hamiltonian(V, dx, mr):
    """Sparse 2D finite-difference Hamiltonian with Dirichlet (hard-wall) boundary."""
    N = V.shape[0]
    t = HBAR2_2ME / (mr * dx * dx)
    n = N * N

    main_diag = 4 * t + V.flatten()
    off_diag_j = -t * np.ones(n - 1)
    # zero the wrap-around coupling at row boundaries (j index wraps every N)
    off_diag_j[np.arange(1, n) % N == 0] = 0
    off_diag_i = -t * np.ones(n - N)

    H = sparse.diags(
        [main_diag, off_diag_j, off_diag_j, off_diag_i, off_diag_i],
        [0, 1, -1, N, -N],
        format="csr",
    )
    return H


def solve_lowest(shape, size_nm, V0, mr, N, k=3):
    """Returns (energies ascending eV, eigenvectors as N x N arrays)."""
    V, dx = build_grid(shape, size_nm, N, V0)
    H = build_hamiltonian(V, dx, mr)
    energies, vecs = eigsh(H, k=k, which="SA")
    order = np.argsort(energies)
    energies = energies[order]
    vecs = vecs[:, order]
    wavefunctions = [vecs[:, i].reshape(N, N) for i in range(k)]
    return energies, wavefunctions, V, dx


def optical_gap(shape, size_nm, V0, material, N=46):
    mat = MATERIALS[material]
    Ee, _, V, dx = solve_lowest(shape, size_nm, V0, mat["me"], N, k=1)
    Eh, _, _, _ = solve_lowest(shape, size_nm, V0, mat["mh"], N, k=1)
    gap = mat["Eg"] + Ee[0] + Eh[0]
    lam = HC / gap
    return {"Ee0": Ee[0], "Eh0": Eh[0], "gap": gap, "lambda_nm": lam, "V": V, "dx": dx}


def wavelength_to_rgb(nm):
    if nm < 380:
        return (80, 0, 120)
    if nm > 750:
        return (50, 0, 0)
    if nm < 440:
        r, g, b = -(nm - 440) / 60, 0, 1
    elif nm < 490:
        r, g, b = 0, (nm - 440) / 50, 1
    elif nm < 510:
        r, g, b = 0, 1, -(nm - 510) / 20
    elif nm < 580:
        r, g, b = (nm - 510) / 70, 1, 0
    elif nm < 645:
        r, g, b = 1, -(nm - 645) / 65, 0
    else:
        r, g, b = 1, 0, 0
    factor = 1.0
    if nm < 420:
        factor = 0.3 + 0.7 * (nm - 380) / 40
    elif nm > 700:
        factor = 0.3 + 0.7 * (750 - nm) / 50
    return (int(255 * r * factor), int(255 * g * factor), int(255 * b * factor))


def region_label(nm):
    if nm < 380: return "Ultraviolet"
    if nm < 450: return "Violet (visible)"
    if nm < 495: return "Blue (visible)"
    if nm < 570: return "Green (visible)"
    if nm < 590: return "Yellow (visible)"
    if nm < 620: return "Orange (visible)"
    if nm <= 750: return "Red (visible)"
    return "Infrared"

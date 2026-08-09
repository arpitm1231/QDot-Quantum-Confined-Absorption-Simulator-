import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st

from qdot_physics import (
    MATERIALS, SHAPES, HC, build_grid, solve_lowest, optical_gap,
    wavelength_to_rgb, region_label,
)
from qdot_surrogate import predict as surrogate_predict, inverse_design, METRICS

# ---------------- page setup / theme ----------------
st.set_page_config(page_title="QDot — Quantum-Confined Absorption Simulator", layout="wide")

BG = "#0a0e1a"; PANEL = "#101828"; BORDER = "#223047"
TEXT = "#e8edf7"; MUTED = "#7c8aa5"; CYAN = "#5fe3d0"; GOLD = "#f2b84b"; VIOLET = "#9b8cff"

st.markdown(f"""
<style>
.stApp {{ background-color:{BG}; color:{TEXT}; }}
section[data-testid="stSidebar"] {{ background-color:{PANEL}; border-right:1px solid {BORDER}; }}
h1,h2,h3 {{ font-family:'Trebuchet MS',sans-serif; }}
.eyebrow {{ font-family:monospace; font-size:11px; letter-spacing:.14em; color:{CYAN}; text-transform:uppercase; }}
.stat-box {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:8px; padding:14px 16px; margin-bottom:8px; }}
.stat-k {{ font-family:monospace; font-size:10px; color:{MUTED}; text-transform:uppercase; letter-spacing:.08em; }}
.stat-v {{ font-size:22px; font-weight:700; color:{TEXT}; }}
.badge {{ display:inline-block; font-family:monospace; font-size:10px; padding:2px 7px; border-radius:4px; background:rgba(155,140,255,.12); color:{VIOLET}; border:1px solid rgba(155,140,255,.35); margin-left:8px; }}
.note {{ font-size:12px; color:{MUTED}; line-height:1.6; border-top:1px solid {BORDER}; padding-top:12px; margin-top:20px; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">2D Finite-Difference Schrödinger Solver · scipy ARPACK (exact) + Neural Surrogate (instant)</div>', unsafe_allow_html=True)
st.title("QDot — Quantum-Confined Absorption Simulator")
st.caption("Python / Streamlit build. Exact solves use scipy's sparse Lanczos eigensolver on the real finite-difference Hamiltonian; the neural surrogate (trained on this solver's own output) powers instant exploration and inverse design.")

# ---------------- sidebar controls ----------------
with st.sidebar:
    st.header("Geometry")
    shape_labels = {"circle": "Circular", "square": "Square", "hexagon": "Hexagonal", "triangle": "Triangular"}
    shape = st.radio("Shape", SHAPES, format_func=lambda s: shape_labels[s], horizontal=True)

    size_nm = st.slider("Dot size (nm)", 2.0, 14.0, 6.0, 0.2)
    V0 = st.slider("Barrier height V₀ (eV)", 0.3, 4.0, 1.5, 0.1)
    N = st.slider("Grid resolution", 30, 60, 46, 2)

    st.header("Material")
    material = st.selectbox("Material", list(MATERIALS.keys()), format_func=lambda k: MATERIALS[k]["label"])

    solve_clicked = st.button("Solve Hamiltonian (exact) →", width='stretch')

# ---------------- live AI estimate (recomputed every rerun — instant) ----------------
live = surrogate_predict(shape, size_nm, V0, material)

st.subheader("Live AI Estimate", anchor=False)
st.markdown(f'<span class="badge">Neural Surrogate</span>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
c1.markdown(f'<div class="stat-box"><div class="stat-k">Predicted gap (instant)</div><div class="stat-v" style="color:{VIOLET}">{live["gap"]:.3f} eV</div></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat-box"><div class="stat-k">Predicted wavelength</div><div class="stat-v" style="color:{VIOLET}">{live["lambda_nm"]:.0f} nm</div></div>', unsafe_allow_html=True)
st.caption(f"Trained on {METRICS['n_train']} solver-generated samples · held-out wavelength MAE ≈ {METRICS['wavelength_mae_nm']} nm (R² {METRICS['r2_Ee0']}/{METRICS['r2_Eh0']}) · updates as sliders move, no solve required")

st.divider()

# ---------------- exact solve ----------------
if "exact_result" not in st.session_state:
    st.session_state.exact_result = None

if solve_clicked:
    with st.spinner("Building Hamiltonian and diagonalizing (scipy ARPACK)…"):
        t0 = time.time()
        mat = MATERIALS[material]
        Ee, psis_e, V, dx = solve_lowest(shape, size_nm, V0, mat["me"], N, k=2)
        Eh, psis_h, _, _ = solve_lowest(shape, size_nm, V0, mat["mh"], N, k=2)
        gap = mat["Eg"] + Ee[0] + Eh[0]
        lam = HC / gap
        elapsed = (time.time() - t0) * 1000
        st.session_state.exact_result = dict(
            shape=shape, size_nm=size_nm, V0=V0, material=material, N=N,
            Ee=Ee, Eh=Eh, psi0=psis_e[0], V=V, gap=gap, lam=lam, elapsed=elapsed,
        )

res = st.session_state.exact_result

st.subheader("Confinement Geometry", anchor=False)
colA, colB = st.columns(2)

def heatmap(ax, data, cmap, title):
    ax.imshow(data.T, origin="lower", cmap=cmap)
    ax.set_title(title, color=MUTED, fontsize=10, fontfamily="monospace")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_visible(False)

fig1, ax1 = plt.subplots(figsize=(4, 4)); fig1.patch.set_facecolor(BG); ax1.set_facecolor("black")
V_preview, _ = build_grid(shape, size_nm, N, V0)
heatmap(ax1, V_preview, mcolors.LinearSegmentedColormap.from_list("v", ["#0a101a", "#ff3d3d"]), "Potential V(x,y)")
colA.pyplot(fig1, width='stretch')

if res is not None:
    fig2, ax2 = plt.subplots(figsize=(4, 4)); fig2.patch.set_facecolor(BG); ax2.set_facecolor("black")
    dens = res["psi0"] ** 2
    heatmap(ax2, dens, mcolors.LinearSegmentedColormap.from_list("p", ["#0a101a", GOLD]), "Electron |ψ₀|² (exact)")
    colB.pyplot(fig2, width='stretch')
else:
    colB.info("Click **Solve Hamiltonian (exact)** in the sidebar to compute the wavefunction density.")

# ---------------- optical readout ----------------
st.subheader("Optical Readout (exact solve)", anchor=False)
if res is None:
    st.info("No exact solve yet — showing the AI live estimate above until you click Solve.")
else:
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="stat-box"><div class="stat-k">Total optical gap</div><div class="stat-v">{res["gap"]:.3f} eV</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-box"><div class="stat-k">Absorption wavelength</div><div class="stat-v">{res["lam"]:.0f} nm</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-box"><div class="stat-k">Spectral region</div><div class="stat-v">{region_label(res["lam"])}</div></div>', unsafe_allow_html=True)

    lc1, lc2 = st.columns(2)
    Ee0, Ee1 = res["Ee"][0], res["Ee"][1]
    Eh0, Eh1 = res["Eh"][0], res["Eh"][1]
    with lc1:
        st.markdown("**Electron confinement levels**")
        st.table({"level": ["E₀ (ground)", "E₁ (1st excited)", "ΔE₀₁"],
                   "energy (eV)": [f"{Ee0:.3f}", f"{Ee1:.3f}", f"{Ee1-Ee0:.3f}"]})
    with lc2:
        st.markdown("**Hole confinement levels**")
        st.table({"level": ["E₀ (ground)", "E₁ (1st excited)", "ΔE₀₁"],
                   "energy (eV)": [f"{Eh0:.3f}", f"{Eh1:.3f}", f"{Eh1-Eh0:.3f}"]})

    # spectral bar
    xs = np.linspace(300, 900, 600)
    colors = np.array([wavelength_to_rgb(x) for x in xs]) / 255.0
    fig3, ax3 = plt.subplots(figsize=(10, 0.9)); fig3.patch.set_facecolor(BG)
    ax3.imshow(colors[np.newaxis, :, :], extent=[300, 900, 0, 1], aspect="auto")
    ax3.axvline(res["lam"], color="white", linewidth=2)
    ax3.set_xlim(300, 900); ax3.set_yticks([])
    ax3.set_xlabel("wavelength (nm)", color=MUTED, fontsize=9, fontfamily="monospace")
    ax3.tick_params(colors=MUTED, labelsize=8)
    for spine in ax3.spines.values(): spine.set_visible(False)
    st.pyplot(fig3, width='stretch')
    st.caption(f"Converged · grid {N}×{N} · Eg(bulk)={MATERIALS[material]['Eg']} eV + confinement · solved in {res['elapsed']:.0f} ms (scipy ARPACK)")

st.divider()

# ---------------- size-tunability sweep ----------------
st.subheader("Size-Tunability Sweep", anchor=False)
st.caption("Uses the neural surrogate for speed, with a few exact scipy solves overlaid as validation checkpoints.")

if st.button("Run size sweep"):
    sizes = np.linspace(max(1.5, size_nm * 0.5), size_nm * 1.9, 12)
    surrogate_lams = [surrogate_predict(shape, sz, V0, material)["lambda_nm"] for sz in sizes]

    checkpoint_sizes = sizes[::4]
    exact_lams = []
    with st.spinner("Running exact validation checkpoints…"):
        for sz in checkpoint_sizes:
            exact_lams.append(optical_gap(shape, sz, V0, material, N=34)["lambda_nm"])

    fig4, ax4 = plt.subplots(figsize=(10, 3.2)); fig4.patch.set_facecolor(BG); ax4.set_facecolor(PANEL)
    ax4.plot(sizes, surrogate_lams, color=GOLD, linewidth=2, label="Surrogate (fast)")
    ax4.scatter(checkpoint_sizes, exact_lams, color=CYAN, zorder=5, s=45, label="Exact solver (checkpoints)")
    ax4.set_xlabel("dot size (nm)", color=MUTED, fontsize=9)
    ax4.set_ylabel("absorption wavelength (nm)", color=MUTED, fontsize=9)
    ax4.tick_params(colors=MUTED, labelsize=8)
    ax4.legend(facecolor=PANEL, edgecolor=BORDER, labelcolor=TEXT, fontsize=9)
    for spine in ax4.spines.values(): spine.set_color(BORDER)
    st.pyplot(fig4, width='stretch')

    max_err = max(abs(s - e) for s, e in zip(
        [surrogate_predict(shape, sz, V0, material)["lambda_nm"] for sz in checkpoint_sizes], exact_lams))
    st.caption(f"Max surrogate-vs-exact deviation at checkpoints: {max_err:.1f} nm — smaller dots blue-shift, confirming quantum confinement.")

st.divider()

# ---------------- inverse design ----------------
st.subheader("Inverse Design", anchor=False)
st.markdown('<span class="badge">AI-Assisted</span>', unsafe_allow_html=True)
st.caption("Pick a target absorption wavelength — the surrogate searches dot size instantly via bisection, instead of re-running the Hamiltonian solver many times.")

target_lambda = st.slider("Target wavelength (nm)", 380, 900, 550, 5)

if "inverse_size" not in st.session_state:
    st.session_state.inverse_size = None

ic1, ic2 = st.columns([1, 1])
with ic1:
    if st.button("Find size →", width='stretch'):
        st.session_state.inverse_size = inverse_design(target_lambda, shape, V0, material)
    if st.session_state.inverse_size is not None:
        if st.button("Verify with exact solver", width='stretch'):
            with st.spinner("Running exact Lanczos solve…"):
                exact = optical_gap(shape, st.session_state.inverse_size, V0, material, N=40)
                err = abs(exact["lambda_nm"] - target_lambda)
                st.success(f"Exact solver at {st.session_state.inverse_size:.2f}nm → {exact['lambda_nm']:.0f}nm (target {target_lambda}nm, off by {err:.0f}nm)")

with ic2:
    if st.session_state.inverse_size is not None:
        st.markdown(f'<div class="stat-box" style="text-align:center"><div class="stat-k">PREDICTED DOT SIZE</div>'
                     f'<div class="stat-v" style="font-size:28px;color:{VIOLET}">{st.session_state.inverse_size:.2f} nm</div>'
                     f'<div class="stat-k">{shape_labels[shape]} · {MATERIALS[material]["label"].split(" —")[0]} · V₀={V0:.1f}eV</div></div>',
                     unsafe_allow_html=True)
    else:
        st.markdown('<div class="stat-box" style="text-align:center;color:${MUTED}">— nm</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="note">
<b>Method:</b> effective-mass approximation, 2D finite-difference Hamiltonian, exact eigenstates via scipy's sparse ARPACK Lanczos solver (validated against the analytical circular infinite-well solution — Bessel-function zeros). Neural surrogate (64→32 hidden units, ~2.8k parameters) trained offline on {METRICS['n_train']} solver-generated samples.<br>
<b>Limits:</b> ignores strain, surface trap states, dielectric confinement, and exciton binding; effective-mass approximation weakens below ~2-3 nm. Fast pre-screening tool, not a replacement for DFT or experiment.
</div>
""", unsafe_allow_html=True)

# QDot — Quantum-Confined Absorption Simulator (Python / Streamlit)

"It's a research and education tool for exploring quantum dot design space fast — a pre-screening step before the expensive stuff (DFT, synthesis, lab validation), not a replacement for it."

Live Demo:https://qdot-quantum-confined.streamlit.app/

## Setup
```
pip install -r requirements.txt
```

## Run
```
streamlit run qdot_app.py
```
Opens at http://localhost:8501

## Files
- `qdot_physics.py` — core solver: 2D finite-difference Hamiltonian, solved exactly via scipy's sparse ARPACK (Lanczos) eigensolver. Validated against the analytical circular infinite-well solution (Bessel function zeros).
- `qdot_surrogate.py` — neural surrogate (loads `surrogate_weights.json`) for instant prediction + inverse design (bisection search).
- `qdot_app.py` — Streamlit UI: shape/size/material controls, exact solve, live AI estimate, size-tunability sweep, inverse design with exact-solver verification.
- `surrogate_weights.json` — trained MLP weights (64→32 hidden units, ~2.8k parameters), trained on 2880 solver-generated samples.
- `generate_dataset.js` / `train_surrogate.py` — scripts used to produce the training data and train the surrogate (not needed to run the app; included for reproducibility).

## Cross-validation
This Python implementation (scipy ARPACK) and the standalone browser/JS version (hand-rolled Lanczos) were built independently and give identical results to 4 decimal places on the same test case (CdSe, 6nm circular dot: Ee0=0.1410 eV, Eh0=0.0457 eV, gap=1.9267 eV, λ=643.5 nm) — strong evidence the physics implementation is correct, not a coincidence of one buggy codebase.

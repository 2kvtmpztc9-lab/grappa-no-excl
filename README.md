# GraPPA without exclusions - a working model

## What it is

An ML-FF based on GraPPA that:
- **predicts** `q, σ, ε` (nonbonded) for each atom **itself**,
- is trained **directly on QM** (`E_ref = E_QM`, without Δ-learning),
- computes nonbonded interactions **over all pairs** with damping (no exclusions during training),
- is **MD-stable** with PME + CustomNonbondedForce.

## Model

**Dataset:** `spice-dipeptide` (677 molecules, SPICE QM level).
**Epochs:** 309 (early stopping).
**Checkpoint:** `checkpoint.ckpt`.

## Metrics on the test set

- `rmse_energies = 2.676` kcal/mol (`ratio_E = 0.128` — the model explains **98.4%** of the QM variance)
- `rmse_gradients = 5.549` kcal/mol/Å (`ratio_G = 0.258`)
- n_confs = 2687

## Parameters (averages)

- `q`: mean = 0.0006, std = 0.342
- `σ`: mean = 2.675 Å, std = 0.673
- `ε`: mean = 0.146 kcal/mol, std = 0.161

## MD (ubiquitin, 1UBQ)

- **System:** `1ubq_system_pme.xml` — 1231 atoms, PME for Coulomb + CustomNonbondedForce for LJ + damping, exceptions 1-2, 1-3.
- **Minimization:** -7450 kcal/mol.
- **10 ps MD:** -8436 kcal/mol — **stable**.

## How to use

```python
from grappa.utils.model_loading_utils import model_from_path
model = model_from_path('checkpoint.ckpt')
model.eval()

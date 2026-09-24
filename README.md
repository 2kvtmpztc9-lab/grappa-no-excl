# GraPPA without Exclusions — A Community Extension

> **Note:** This is **not** an official GraPPA release. It is a **community extension** built **on top of** the original [GraPPA](https://github.com/graeter-group/grappa) codebase by the Gräter Group. This repository contains only the **modified and added files** needed to reproduce our experiments. The original GraPPA code is **not** included here — you need to install it separately.

## Why we built this

The original GraPPA is a Δ-learning framework: it predicts **bonded** parameters, while **nonbonded** interactions (`q`, `σ`, `ε`) and **exclusions** are taken from a classical force field (amber99, charmm36, openff, etc.). This works well for many applications, but it creates a systematic problem in **QM/MM** setups.

In classical force fields, **exclusions** are used to skip nonbonded interactions between atoms that are already connected through bonds (1-2), angles (1-3), and sometimes torsions (1-4). This is a **technical trick** to avoid the singularity of the Lennard-Jones potential at short distances. Without exclusions, the LJ energy for a bonded pair diverges (`10^6` kcal/mol or more).

However, **QM** doesn't use exclusions — it computes everything from first principles. So when you train a model on the difference `E_QM − E_MM`, that difference contains a **systematic error** coming from the exclusions. The model learns to compensate for this error, which leads to **artifacts** when you transfer the model to new molecules.

Our goal was to build an ML force field that **doesn't need exclusions** in the first place.

## Our approach

We modified GraPPA in three key ways:

1. **Nonbonded head**: We added a new output to the model that predicts `q`, `σ`, and `ε` for every atom, directly from its local environment. The model no longer relies on a classical force field for nonbonded parameters.

2. **Direct QM training**: Instead of learning the difference `E_QM − E_MM`, we set `E_ref = E_QM` and train the model on the full QM energy and gradients. This is done via a small monkey-patch that overrides `Dataset.create_reference`.

3. **Damping**: To avoid the LJ singularity without exclusions, we use a damping function:

   
## Files we changed in the original GraPPA

If you want to apply our changes to your own GraPPA installation, here's the full list:

| File | Change |
|------|--------|
| `src/grappa/models/nonbonded.py` | **NEW** — `WriteNonbondedParameters` class |
| `src/grappa/models/interaction_parameters.py` | Added `nonbonded_hidden_feats` parameter and `nonbonded_writer` |
| `src/grappa/models/grappa.py` | Added `nonbonded_hidden_feats` to `GrappaModel` |
| `src/grappa/models/energy.py` | Added `_nonbonded_energy` method and its call in `forward` |
| `src/grappa/training/loss.py` | Added regularization for `σ` and `ε` |
| `src/grappa/utils/dgl_utils.py` | Fixed double-offset bug in `batch()` |
| `src/grappa/data/dataset.py` | Handled `ref_terms=[]` in `create_reference` |

We also use a **monkey-patch** (`experiment.py`) that overrides `Dataset.create_reference` so that `energy_ref = energy_qm` instead of `energy_qm − energy_nonbonded`. This avoids modifying the original GraPPA code more than necessary.

## How to use

### 1. Install the original GraPPA

```bash
git clone https://github.com/graeter-group/grappa.git
cd grappa
pip install -e .

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

   ```
   f_damp(r) = 1 / (1 + exp(-α · (r/r0 − 1)))
   ```

   with `r0 = 0.3 nm` and `α = 20`. This smoothly suppresses the LJ and Coulomb contributions at short distances while leaving them unchanged for `r > 5 Å`.

## What's in this repository

```
grappa_no_excl_package/
├── src/grappa/                    # Modified GraPPA code (only changed files)
│   ├── models/
│   │   ├── nonbonded.py          # NEW: WriteNonbondedParameters head
│   │   ├── interaction_parameters.py  # MODIFIED: wired up the head
│   │   ├── grappa.py             # MODIFIED: added nonbonded_hidden_feats
│   │   └── energy.py             # MODIFIED: added _nonbonded_energy
│   ├── training/
│   │   └── loss.py               # MODIFIED: added σ/ε regularization
│   ├── utils/
│   │   └── dgl_utils.py          # MODIFIED: fixed batch mutation bug
│   └── data/
│       └── dataset.py            # MODIFIED: handled ref_terms=[] edge case
├── configs/
│   ├── train_no_excl.yaml         # Main training config
│   └── experiment.yaml            # Experiment hyperparameters
├── experiments/
│   ├── train_no_excl.py           # Training script (with monkey-patch import)
│   └── evaluate_no_excl.py        # Evaluation script
├── scripts/
│   └── build_openmm_system_pme.py # Export to OpenMM with PME
├── patches/                       # Diff patches for each modified file
│   ├── interaction_parameters.py.patch
│   ├── grappa.py.patch
│   ├── energy.py.patch
│   ├── loss.py.patch
│   ├── dgl_utils.py.patch
│   ├── dataset.py.patch
│   └── nonbonded.py.new           # New file, copied in full
├── experiment.py                  # Monkey-patch: energy_ref = energy_qm
├── grappa-no-excl-spice/          # Trained model + MD files
│   ├── checkpoint.ckpt            # Model weights (epoch 309, early_stopped)
│   ├── config.yaml                # Training config
│   ├── split.json                 # Train/val/test split
│   ├── 1ubq_system_pme.xml        # OpenMM System (MD-ready)
│   ├── 1ubq_with_H.pdb            # Ubiquitin with hydrogens
│   ├── 1ubq_params.npz            # Predicted parameters
│   └── README.md                  # Model-specific readme
└── README.md                      # This file
```

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
```

### 2. Apply our changes

Copy the files from `src/grappa/` in this repository into your GraPPA installation, **overwriting** the originals. Or, if you prefer, use `git apply` with the patches in `patches/`:

```bash
cd /path/to/grappa
for patch in /path/to/grappa_no_excl_package/patches/*.patch; do
    git apply "$patch"
done
cp /path/to/grappa_no_excl_package/patches/nonbonded.py.new \
   src/grappa/models/nonbonded.py
```

### 3. Install the monkey-patch

Copy `experiment.py` to a directory in your `PYTHONPATH`, e.g.:

```bash
cp experiment.py ~/my_project/
```

### 4. Run training or evaluation

```bash
cd ~/grappa
PYTHONPATH=/path/to/this/repo python experiments/train_no_excl.py
```

## Results

We trained on `spice-dipeptide` (677 molecules, SPICE QM level) and `spice-pubchem` (14,110 molecules, SPICE QM level). Metrics on the test set:

| Dataset | Epochs | `rmse_E` (kcal/mol) | `ratio_E` | `rmse_G` (kcal/mol/Å) | `ratio_G` |
|---------|--------|---------------------|-----------|------------------------|-----------|
| spice-dipeptide | 309 | 2.676 | **0.128** | 5.549 | 0.258 |
| spice-pubchem | 469 | 2.688 | **0.136** | 6.170 | 0.265 |

For comparison, the published GraPPA-1.4.0 (Δ-learning, with exclusions) reaches `rmse_E ≈ 2.3` kcal/mol on `spice-pubchem`. Our model is close to that, **without using exclusions** during training.

### MD stability

We tested MD on ubiquitin (1UBQ, 1231 atoms). The system uses:
- `NonbondedForce` with **PME** for Coulomb (with exceptions 1-2, 1-3),
- `CustomNonbondedForce` with **damping** for LJ (with the same exceptions),
- our bonded parameters from the model.

After energy minimization (`-7450` kcal/mol), 10 ps MD runs stably at `-8436` kcal/mol. Without PME, the system crashes (`NaN`) within a few ps.

### Model parameters (averages)

- `q`: mean = 0.0006, std = 0.342
- `σ`: mean = 2.675 Å, std = 0.673
- `ε`: mean = 0.146 kcal/mol, std = 0.161

## Limitations and honest caveats

- **Exclusions for 1-2 and 1-3 are still present** in the MD system. We could not remove them entirely because PME cannot handle such short distances, and Coulomb for 1-3 pairs without damping dominates the energy (34,000 kcal/mol for ubiquitin). This is a **compromise**: 1-4 and beyond use our learned `q, σ, ε` without scaling, while 1-2 and 1-3 are excluded as in AMBER.
- **Attempts to learn Coulomb damping per-atom** (`α_coul`, `r0_coul`) failed: the loss explodes (`12,554` instead of `~20`), and the model gets stuck at the bounds of the damping parameters. We're still working on this.
- **The model was trained on peptides**, not on general organic molecules. It works well on `spice-dipeptide` and `spice-pubchem`, but we have not validated it on proteins or nucleic acids.
- **This is a research prototype**, not a production-ready force field.

## What we're working on next

- **Learned Coulomb damping** (per-atom `α_coul`, `r0_coul`) — currently unstable.
- **Soft-core LJ** for a more stable MD without exclusions for 1-3.
- **Extensions to larger datasets** (`spice-des-monomers`, `gen2`).
- **Comparison with baseline** GraPPA-1.4.0 on the same test set.

## Acknowledgements

This work builds directly on [GraPPA](https://github.com/graeter-group/grappa) by the Gräter Group. We thank them for releasing their code under a permissive license, which made this extension possible. This repository contains **only our modifications** — the original GraPPA code is not redistributed here.

## License

Our modifications follow the same license as the original GraPPA. See the original repository for details.

## Contact

For questions about this extension, please open an issue in this repository. For questions about the original GraPPA, please refer to the [original repository](https://github.com/graeter-group/grappa).
We also use a **monkey-patch** (`experiment.py`) that overrides `Dataset.create_reference` so that `energy_ref = energy_qm` instead of `energy_qm − energy_nonbonded`. This avoids modifying the original GraPPA code more than necessary.

## How to use

### 1. Install the original GraPPA

```bash
git clone https://github.com/graeter-group/grappa.git
cd grappa
pip install -e .

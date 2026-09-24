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

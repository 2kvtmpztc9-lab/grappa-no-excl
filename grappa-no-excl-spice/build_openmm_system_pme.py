"""OpenMM System: PME для Coulomb + CustomNonbondedForce для LJ (с одинаковыми exceptions)."""
import sys
from pathlib import Path

sys.path.insert(0, '/home/domain/kiraluk/grappa_zero')
import experiment

import numpy as np
import torch
import openmm
from openmm import (
    System, NonbondedForce, CustomNonbondedForce,
    HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce,
    XmlSerializer, unit,
)
from openmm.app import PDBFile, ForceField
from pdbfixer import PDBFixer
from grappa.data import Molecule, Parameters
from grappa.utils.model_loading_utils import model_from_path


KJ_PER_KCAL = 4.184
NM_PER_A = 0.1
R0_NM = 0.3
ALPHA = 20.0


def main():
    fixer = PDBFixer(filename='1ubq.pdb')
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    fixer.removeHeterogens(keepWater=False)

    ff = ForceField('amber99sbildn.xml')
    amber_system = ff.createSystem(fixer.topology)

    mol = Molecule.from_openmm_system(
        openmm_system=amber_system,
        openmm_topology=fixer.topology,
        partial_charges=None,
    )
    print(f'Molecule: {mol}')

    ckpt_root = Path('ckpt/grappa-experiment/grappa-no-excl-spice')
    runs = sorted(ckpt_root.glob('*/'), key=lambda p: p.stat().st_mtime, reverse=True)
    ckpts = list(runs[0].glob('epoch:*.ckpt'))
    ckpt = max(ckpts, key=lambda p: int(p.stem.split('epoch:')[1].split('-')[0]))
    model = model_from_path(str(ckpt))
    model.eval()

    g = mol.to_dgl()
    with torch.no_grad():
        g = model(g)

    q = g.nodes['n1'].data['q'].cpu().numpy()
    sigma = g.nodes['n1'].data['sigma'].cpu().numpy()
    epsilon = g.nodes['n1'].data['epsilon'].cpu().numpy()
    params = Parameters.from_dgl(g, check_eq_values=False)

    n_atoms = len(list(fixer.topology.atoms()))
    new_system = System()
    for atom in fixer.topology.atoms():
        mass = atom.element.mass.value_in_unit(unit.dalton)
        new_system.addParticle(mass)

    excluded_pairs = set()
    for bond in fixer.topology.bonds():
        i, j = bond[0].index, bond[1].index
        excluded_pairs.add((min(i, j), max(i, j)))

    print(f'Excluded pairs: {len(excluded_pairs)}')

    nbf = NonbondedForce()
    nbf.setNonbondedMethod(NonbondedForce.PME)
    nbf.setCutoffDistance(1.0)
    nbf.setEwaldErrorTolerance(0.0005)
    for i in range(n_atoms):
        nbf.addParticle(
            float(q[i]) * unit.elementary_charge,
            1.0 * unit.nanometer,
            0.0 * unit.kilojoule_per_mole,
        )
    for i, j in excluded_pairs:
        nbf.addException(i, j,
            chargeProd=0.0 * unit.elementary_charge**2,
            sigma=1.0 * unit.nanometer,
            epsilon=0.0 * unit.kilojoule_per_mole,
            replace=True)
    new_system.addForce(nbf)

    cnb = CustomNonbondedForce(
        f"4*sqrt(epsilon1*epsilon2)*"
        f"((0.5*(sigma1+sigma2)/r)^12 - (0.5*(sigma1+sigma2)/r)^6) * "
        f"(1/(1+exp(-{ALPHA}*(r/{R0_NM}-1))))"
    )
    cnb.addPerParticleParameter("sigma")
    cnb.addPerParticleParameter("epsilon")
    for i in range(n_atoms):
        cnb.addParticle([
            float(sigma[i]) * NM_PER_A,
            float(epsilon[i]) * KJ_PER_KCAL,
        ])
    cnb.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    cnb.setCutoffDistance(1.0)
    for i, j in excluded_pairs:
        cnb.addExclusion(i, j)
    new_system.addForce(cnb)

    bond_force = HarmonicBondForce()
    for i, bond in enumerate(params.bonds):
        bond_force.addBond(
            int(bond[0]), int(bond[1]),
            float(params.bond_eq[i]) * NM_PER_A,
            float(params.bond_k[i]) * KJ_PER_KCAL / (NM_PER_A ** 2),
        )
    new_system.addForce(bond_force)

    angle_force = HarmonicAngleForce()
    for i, angle in enumerate(params.angles):
        angle_force.addAngle(
            int(angle[0]), int(angle[1]), int(angle[2]),
            float(params.angle_eq[i]),
            float(params.angle_k[i]) * KJ_PER_KCAL,
        )
    new_system.addForce(angle_force)

    proper_force = PeriodicTorsionForce()
    for i, proper in enumerate(params.propers):
        for n in range(params.proper_ks.shape[1]):
            k = float(params.proper_ks[i, n])
            if k == 0:
                continue
            proper_force.addTorsion(
                int(proper[0]), int(proper[1]), int(proper[2]), int(proper[3]),
                n + 1, float(params.proper_phases[i, n]), k * KJ_PER_KCAL,
            )
    new_system.addForce(proper_force)

    improper_force = PeriodicTorsionForce()
    for i, improper in enumerate(params.impropers):
        for n in range(params.improper_ks.shape[1]):
            k = float(params.improper_ks[i, n])
            if k == 0:
                continue
            improper_force.addTorsion(
                int(improper[0]), int(improper[1]), int(improper[2]), int(improper[3]),
                n + 1, float(params.improper_phases[i, n]), k * KJ_PER_KCAL,
            )
    new_system.addForce(improper_force)

    xml_path = Path('1ubq_system_pme.xml')
    with open(xml_path, 'w') as f:
        f.write(XmlSerializer.serialize(new_system))
    print(f'Saved: {xml_path.absolute()}')

    from openmm.app import PDBFile as PDBWriter
    with open('1ubq_with_H.pdb', 'w') as f:
        PDBWriter.writeFile(fixer.topology, fixer.positions, f)
    print('Saved: 1ubq_with_H.pdb')


if __name__ == '__main__':
    main()

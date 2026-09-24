"""
Monkey-patch GraPPA: всегда energy_ref = energy_qm.
"""
import torch
from grappa.data.dataset import Dataset as _Dataset


def _patched_create_reference(self, ref_terms=["nonbonded"], ff_lookup={}, cleanup=False):
    for g in self.graphs:
        g.nodes['g'].data['energy_ref'] = g.nodes['g'].data['energy_qm'].clone()
        if 'gradient_qm' in g.nodes['n1'].data.keys():
            g.nodes['n1'].data['gradient_ref'] = g.nodes['n1'].data['gradient_qm'].clone()

        if cleanup:
            for k in list(g.nodes['g'].data.keys()):
                if 'energy_' in k and k not in ['energy_ref', 'energy_qm']:
                    del g.nodes['g'].data[k]
            for k in list(g.nodes['n1'].data.keys()):
                if 'gradient_' in k and k not in ['gradient_ref', 'gradient_qm']:
                    del g.nodes['n1'].data[k]
    return


_original_create_reference = _Dataset.create_reference
_Dataset.create_reference = _patched_create_reference
print("[monkey-patch] Dataset.create_reference patched: energy_ref = energy_qm (always)")

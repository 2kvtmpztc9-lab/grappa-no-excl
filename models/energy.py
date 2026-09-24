import torch
import dgl
from grappa.utils.dgl_utils import grad_available

from grappa.models.internal_coordinates import InternalCoordinates
import copy


def torsion_energy(k, angle, offset=True):
    """
    returns a tensor of shape tuples x confs containing the energy contributions of each torsion angle individually.

    implements
    sum_n k_n cos(n*phi) (+ |k_n| if offset=True)

    shape of k: tuples x periodicity
    shape of angle: tuples x confs
    phases are all zero
    k[some_tuple] must be ordered with increasing periodicity, starting at 1
    angle must be in radians
    if offset if False, implements \sum_n k_n cos(n*phi) (i.e. without offset)
    """
    max_periodicity = k.shape[1]
    periodicity = torch.tensor(range(1, max_periodicity+1), device=k.device, dtype=torch.float32).unsqueeze(dim=0).unsqueeze(dim=-1)
    angle = angle.unsqueeze(dim=1)
    k = k.unsqueeze(dim=-1)

    if not offset:
        energy = k*torch.cos(periodicity*angle)
    else:
        energy = torch.abs(k) + k*torch.cos(periodicity*angle)

    energy = energy.sum(dim=1)
    return energy


def harmonic_energy(k, eq, distances):
    """
    returns a tensor of shape tuples x confs containing the energy contributions of each tuple (bond/angle) individually.
    implements
    0.5 * k * (distances - eq)^2
    """
    if len(k.shape) != 1:
        raise ValueError(f"k must be a 1d tensor, but has shape {k.shape}")
    energy = k.unsqueeze(dim=-1)*torch.square(distances-eq.unsqueeze(dim=-1))
    return 0.5 * energy


def pool_energy(g, energies, term, suffix):
    """
    Given a tensor of energy contributions of shape tuples x confs, returns the energy pooled over the tuple dimension.
    """
    if not energies.shape[0] == g.num_nodes(term):
        raise ValueError(f"shape of energies {energies.shape} does not match number of nodes {g.num_nodes(term)}")

    g.nodes[term].data['unpooled_energy'+suffix] = energies
    pooled_energies = dgl.readout_nodes(g, op='sum', ntype=term, feat='unpooled_energy'+suffix)
    return pooled_energies


class Energy(torch.nn.Module):
    def __init__(self, terms:list=["bond", "angle", "torsion", "improper"], suffix:str="", offset_torsion:bool=False,
                 write_suffix=None, gradients:bool=True, gradient_contributions:bool=False):
        super().__init__()

        if not isinstance(terms, list):
            raise ValueError("terms must be a list")
        self.offset_torsion = offset_torsion
        self.suffix = suffix
        self.write_suffix = write_suffix if not write_suffix is None else suffix
        self.gradients = gradients
        self.gradient_contributions = gradient_contributions
        self.geom = InternalCoordinates()

        self.TERM_TO_LEVEL = {
            "bond": "n2",
            "angle": "n3",
            "proper": "n4",
            "torsion": "n4",
            "improper": "n4_improper"
        }
        self.LEVEL_TO_TERM = {v: k for k, v in self.TERM_TO_LEVEL.items()}
        self.LEVEL_TO_TERM["n4"] = "proper"
        self.terms = [self.TERM_TO_LEVEL[t] for t in terms]

    # NONBONDED
    def _nonbonded_energy(self, g, coulomb_const=332.0636):
        """
        Nonbonded энергия по всем парам i<j.
        LJ — с фиксированным damping (r0=0.3 nm, alpha=20).
        Coulomb — с ОБУЧАЕМЫМ damping (alpha_coul, r0_coul per-atom).
        """
        xyz = g.nodes["n1"].data["xyz"]              # (n_atoms, n_confs, 3)
        q = g.nodes["n1"].data["q" + self.suffix]
        sigma = g.nodes["n1"].data["sigma" + self.suffix]
        epsilon = g.nodes["n1"].data["epsilon" + self.suffix]
        alpha_coul = g.nodes["n1"].data["alpha_coul" + self.suffix]
        r0_coul = g.nodes["n1"].data["r0_coul" + self.suffix]

        # LJ damping
        R0_LJ = 0.3      # nm (3 Å)
        ALPHA_LJ = 20.0

        n_confs = xyz.shape[1]
        batch_sizes = g.batch_num_nodes('n1').tolist()
        n_batch = len(batch_sizes)

        E_total = torch.zeros((n_batch, n_confs), device=xyz.device, dtype=xyz.dtype)

        offset = 0
        for b, n in enumerate(batch_sizes):
            if n < 2:
                offset += n
                continue

            xyz_b = xyz[offset:offset + n]
            q_b = q[offset:offset + n]
            sigma_b = sigma[offset:offset + n]
            epsilon_b = epsilon[offset:offset + n]
            alpha_coul_b = alpha_coul[offset:offset + n]
            r0_coul_b = r0_coul[offset:offset + n]
            offset += n

            # Все пары i < j:
            i, j = torch.triu_indices(n, n, offset=1, device=xyz.device)
            diff = xyz_b[i] - xyz_b[j]                        # (n_pairs, n_confs, 3)
            r = torch.norm(diff, dim=-1) + 1e-8               # (n_pairs, n_confs)

            # LJ
            sigma_ij = 0.5 * (sigma_b[i] + sigma_b[j])        # (n_pairs,)
            epsilon_ij = torch.sqrt(epsilon_b[i] * epsilon_b[j])
            sr = sigma_ij.unsqueeze(-1) / r                   # (n_pairs, n_confs)
            lj = 4 * epsilon_ij.unsqueeze(-1) * (sr ** 12 - sr ** 6)
            f_damp_lj = 1.0 / (1.0 + torch.exp(-ALPHA_LJ * (r / R0_LJ - 1.0)))

            #damping 
            q_ij = q_b[i] * q_b[j]                            # (n_pairs,)
            alpha_coul_ij = 0.5 * (alpha_coul_b[i] + alpha_coul_b[j])   # (n_pairs,)
            r0_coul_ij = 0.5 * (r0_coul_b[i] + r0_coul_b[j])            # (n_pairs,)
            f_damp_coul = 1.0 / (1.0 + torch.exp(
                -alpha_coul_ij.unsqueeze(-1) * (r / r0_coul_ij.unsqueeze(-1) - 1.0)
            ))
            coul = coulomb_const * q_ij.unsqueeze(-1) / r * f_damp_coul

            E_total[b] = (lj * f_damp_lj + coul).sum(dim=0)

        return E_total

    
    def forward(self, g):
        if self.gradient_contributions and not self.gradients:
            raise ValueError("Gradient contributions cannot be calculated if gradients are not enabled.")

        grad_enabled = copy.deepcopy(torch.is_grad_enabled())
        if not grad_enabled and self.gradients:
            torch.set_grad_enabled(True)

        if not "xyz" in g.nodes["n1"].data.keys():
            raise ValueError("xyz coordinates must be stored in g.nodes['n1'].data['xyz']")

        if self.gradients:
            with torch.enable_grad():
                g.nodes["n1"].data["xyz"].requires_grad = True

        # координаты
        g = self.geom(g)

        num_confs = g.nodes['n1'].data["xyz"].shape[1]
        num_batch = g.num_nodes("g")

        energy = torch.zeros((num_batch, num_confs), device=g.nodes['n1'].data["xyz"].device)

        for term in self.terms:
            termname = self.LEVEL_TO_TERM[term]
            if not term in g.ntypes:
                raise ValueError(f"term {term} not in g.ntypes")

            contrib, tuple_energies = Energy.get_energy_contribution(g, term=term, suffix=self.suffix, offset_torsion=self.offset_torsion)
            if not contrib is None:
                if self.gradient_contributions and grad_available():
                    if contrib.shape[0] > 0 and not torch.all(contrib == 0):
                        grad = torch.autograd.grad(contrib.sum(), g.nodes["n1"].data["xyz"], retain_graph=True, create_graph=True, allow_unused=True)[0]
                    else:
                        grad = torch.zeros_like(g.nodes["n1"].data["xyz"])
                    g.nodes["n1"].data["gradient_" + self.write_suffix + termname] = grad

                energy += contrib
                g.nodes["g"].data["energy_" + self.write_suffix + termname] = contrib.detach()
                g.nodes[term].data["energy" + self.write_suffix] = tuple_energies

        # nonbonded 
        if "q" + self.suffix in g.nodes["n1"].data:
            contrib_nb = self._nonbonded_energy(g)
            energy = energy + contrib_nb
            g.nodes["g"].data["energy_nonbonded" + self.write_suffix] = contrib_nb.detach()

            if self.gradients and grad_available():
                grad_nb = torch.autograd.grad(
                    contrib_nb.sum(), g.nodes["n1"].data["xyz"],
                    retain_graph=True, create_graph=True, allow_unused=True
                )[0]
                g.nodes["n1"].data["gradient_nonbonded" + self.write_suffix] = grad_nb

        g.nodes["g"].data["energy" + self.write_suffix] = energy

        # gradient 
        if self.gradients and grad_available():
            grad = torch.autograd.grad(energy.sum(), g.nodes["n1"].data["xyz"], retain_graph=True, create_graph=True, allow_unused=True)[0]
            g.nodes["n1"].data["gradient" + self.write_suffix] = grad

        if self.gradients:
            torch.set_grad_enabled(grad_enabled)

        return g

    @staticmethod
    def get_energy_contribution(g, term, suffix, offset_torsion=True):
        if term not in g.ntypes:
            return None, None
        if "k" + suffix not in g.nodes[term].data.keys():
            raise RuntimeError(f"{term} has no k{suffix} attribute")

        k = g.nodes[term].data["k" + suffix]
        dof_data = g.nodes[term].data["x"]

        if term in ["n2", "n3"]:
            eq = g.nodes[term].data["eq" + suffix]
            energies = harmonic_energy(k=k, eq=eq, distances=dof_data)
            en = pool_energy(g=g, energies=energies, term=term, suffix=suffix)

        if term in ["n4", "n4_improper"]:
            energies = torsion_energy(k=k, angle=dof_data, offset=offset_torsion)
            en = pool_energy(g=g, energies=energies, term=term, suffix=suffix)

        return en, energies

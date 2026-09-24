import torch
from grappa.models.final_layer import ToRange, ToPositive


class ToRangeWithMin(torch.nn.Module):
    def __init__(self, min_, max_, std=1.0):
        super().__init__()
        self.register_buffer('min_', torch.tensor(float(min_)))
        self.register_buffer('range_', torch.tensor(float(max_ - min_)))
        self.register_buffer('std_', torch.tensor(float(std)))
    def forward(self, x):
        sigmoid_x = torch.sigmoid(self.std_ * x)
        return self.min_ + self.range_ * sigmoid_x

class WriteNonbondedParameters(torch.nn.Module):
    def __init__(self, rep_feats=256, hidden_feats=256, suffix="",
                 q_mean=0.0, q_std=0.5,
                 sigma_min=2.0, sigma_max=4.5,
                 epsilon_min=0.02, epsilon_max=0.5,
                 alpha_coul_min=1.0, alpha_coul_max=50.0,
                 r0_coul_min=0.1, r0_coul_max=0.5):
        super().__init__()
        self.suffix = suffix
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(rep_feats, hidden_feats),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_feats, hidden_feats),
            torch.nn.ELU(),
            torch.nn.Linear(hidden_feats, 5),   # <-- 5 вместо 3
        )
        self.register_buffer("q_mean", torch.tensor(q_mean, dtype=torch.float32))
        self.register_buffer("q_std", torch.tensor(q_std, dtype=torch.float32))
        self.to_sigma = ToRangeWithMin(min_=sigma_min, max_=sigma_max, std=1.0)
        self.to_epsilon = ToRangeWithMin(min_=epsilon_min, max_=epsilon_max, std=1.0)
        self.to_alpha_coul = ToRangeWithMin(min_=alpha_coul_min, max_=alpha_coul_max, std=1.0)
        self.to_r0_coul = ToRangeWithMin(min_=r0_coul_min, max_=r0_coul_max, std=1.0)

    def forward(self, g):
        h = g.nodes["n1"].data["h"]
        out = self.mlp(h)
        q = self.q_mean + self.q_std * out[:, 0]
        sigma = self.to_sigma(out[:, 1])
        epsilon = self.to_epsilon(out[:, 2])
        alpha_coul = self.to_alpha_coul(out[:, 3])
        r0_coul = self.to_r0_coul(out[:, 4])
        g.nodes["n1"].data["q" + self.suffix] = q
        g.nodes["n1"].data["sigma" + self.suffix] = sigma
        g.nodes["n1"].data["epsilon" + self.suffix] = epsilon
        g.nodes["n1"].data["alpha_coul" + self.suffix] = alpha_coul
        g.nodes["n1"].data["r0_coul" + self.suffix] = r0_coul
        return g

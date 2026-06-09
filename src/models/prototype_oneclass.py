import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeOneClass(nn.Module):
    """
    Prototype-based one-class scorer on top of learned latent features.
    - input: feature Z shape [B, D]
    - output: minimum squared distance to any prototype
    """

    def __init__(self, in_dim, hidden_dim=128, rep_dim=64, num_prototypes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.rep_dim = int(rep_dim)
        self.num_prototypes = max(1, int(num_prototypes))
        self.prototypes = nn.Parameter(torch.empty(self.num_prototypes, self.rep_dim))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    @staticmethod
    def pairwise_dist2(X, Y):
        x_norm = (X * X).sum(dim=1, keepdim=True)
        y_norm = (Y * Y).sum(dim=1).unsqueeze(0)
        return torch.clamp(x_norm + y_norm - 2.0 * (X @ Y.T), min=0.0)

    def encode(self, Z):
        return self.net(Z)

    @torch.no_grad()
    def init_prototypes(self, Z):
        H = self.encode(Z)
        if H.shape[0] == 0:
            raise ValueError("PrototypeOneClass.init_prototypes requires at least one sample.")

        if self.num_prototypes == 1:
            self.prototypes.copy_(H.mean(dim=0, keepdim=True))
            return

        selected = []
        mean = H.mean(dim=0, keepdim=True)
        dist_to_mean = self.pairwise_dist2(H, mean).squeeze(1)
        first_idx = int(torch.argmax(dist_to_mean).item())
        selected.append(H[first_idx])

        min_dist2 = self.pairwise_dist2(H, selected[0].unsqueeze(0)).squeeze(1)
        while len(selected) < self.num_prototypes:
            next_idx = int(torch.argmax(min_dist2).item())
            selected.append(H[next_idx])
            latest = self.pairwise_dist2(H, selected[-1].unsqueeze(0)).squeeze(1)
            min_dist2 = torch.minimum(min_dist2, latest)

        self.prototypes.copy_(torch.stack(selected, dim=0))

    def forward(self, Z):
        H = self.encode(Z)
        dist2 = self.pairwise_dist2(H, self.prototypes)
        return dist2.min(dim=1).values

    def loss(self, Z, separation_weight=0.1, separation_margin=1.0):
        H = self.encode(Z)
        dist2 = self.pairwise_dist2(H, self.prototypes)
        min_dist2 = dist2.min(dim=1).values
        loss = min_dist2.mean()

        sep_penalty = min_dist2.new_tensor(0.0)
        if self.num_prototypes > 1 and separation_weight > 0.0:
            proto_dist2 = self.pairwise_dist2(self.prototypes, self.prototypes)
            mask = ~torch.eye(self.num_prototypes, device=proto_dist2.device, dtype=torch.bool)
            margin2 = float(separation_margin) ** 2
            sep_penalty = F.relu(margin2 - proto_dist2[mask]).mean()
            loss = loss + float(separation_weight) * sep_penalty

        return loss, min_dist2, sep_penalty

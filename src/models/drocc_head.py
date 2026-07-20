import torch
import torch.nn as nn


class DROCCHead(nn.Module):
    """
    DROCC-style latent one-class scorer.
    - encoder maps input features into a compact latent space
    - classifier predicts anomaly logits from centered latent codes
    - adversarial negatives are synthesized in a radius shell around normal latents
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, rep_dim: int = 64):
        super().__init__()
        self.rep_dim = int(rep_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rep_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(rep_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("train_center", torch.zeros(self.rep_dim))

    def encode(self, Z: torch.Tensor) -> torch.Tensor:
        return self.encoder(Z)

    @torch.no_grad()
    def init_center(self, Z: torch.Tensor, eps: float = 1e-3):
        self.eval()
        out = self.encode(Z)
        c = out.mean(dim=0)
        c[(c.abs() < eps) & (c < 0)] = -eps
        c[(c.abs() < eps) & (c > 0)] = eps
        self.train_center.copy_(c)

    def compactness_loss(self, Z: torch.Tensor):
        h = self.encode(Z)
        dist2 = ((h - self.train_center) ** 2).sum(dim=1)
        return dist2.mean(), dist2, h

    def logits_from_latent(self, H: torch.Tensor) -> torch.Tensor:
        centered = H - self.train_center
        return self.classifier(centered).squeeze(1)

    def random_shell_latents(self, H: torch.Tensor, radius: float, gamma: float) -> torch.Tensor:
        radius = float(max(radius, 1e-4))
        max_radius = max(radius, radius * float(max(gamma, 1.0)))
        noise = torch.randn_like(H)
        noise = noise / noise.norm(dim=1, keepdim=True).clamp_min(1e-8)
        shell_scale = torch.empty(H.shape[0], 1, device=H.device, dtype=H.dtype).uniform_(radius, max_radius)
        return H + noise * shell_scale

    def project_to_shell(self, H: torch.Tensor, H_adv: torch.Tensor, radius: float, gamma: float) -> torch.Tensor:
        radius = float(max(radius, 1e-4))
        max_radius = max(radius, radius * float(max(gamma, 1.0)))
        delta = H_adv - H
        norm = delta.norm(dim=1, keepdim=True).clamp_min(1e-8)
        projected_norm = norm.clamp(min=radius, max=max_radius)
        return H + delta / norm * projected_norm

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        return self.logits_from_latent(self.encode(Z))

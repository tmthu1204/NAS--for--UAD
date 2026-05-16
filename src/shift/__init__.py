"""Domain-shift utilities based on latent representations."""

from .pad import compute_pad_from_latents, proxy_a_distance_from_accuracy
from .ts_jepa import (
    TSJepaConfig,
    TSJepaModel,
    extract_jepa_features,
    train_ts_jepa,
)

__all__ = [
    "TSJepaConfig",
    "TSJepaModel",
    "compute_pad_from_latents",
    "extract_jepa_features",
    "proxy_a_distance_from_accuracy",
    "train_ts_jepa",
]

"""Model-family helpers for pluggable paper backbones."""

from .omni_anomaly import (
    OmniArchConfig,
    OmniAnomalyModel,
    get_fixed_paper_omni_arch,
    sample_omni_arch,
    score_omni_series,
    train_omni_source,
    validate_omni_on_series,
)
from .usad import (
    UsadArchConfig,
    UsadModel,
    get_fixed_paper_usad_arch,
    sample_usad_arch,
    score_usad_series,
    train_usad_source,
    validate_usad_on_series,
)
from .dagmm import (
    DagmmArchConfig,
    DagmmModel,
    get_fixed_paper_dagmm_arch,
    sample_dagmm_arch,
    score_dagmm_series,
    train_dagmm_source,
    validate_dagmm_on_series,
)
from .tranad import (
    TranADArchConfig,
    TranADModel,
    get_fixed_paper_tranad_arch,
    sample_tranad_arch,
    score_tranad_series,
    train_tranad_source,
    validate_tranad_on_series,
)

__all__ = [
    "OmniArchConfig",
    "OmniAnomalyModel",
    "get_fixed_paper_omni_arch",
    "sample_omni_arch",
    "score_omni_series",
    "train_omni_source",
    "validate_omni_on_series",
    "UsadArchConfig",
    "UsadModel",
    "get_fixed_paper_usad_arch",
    "sample_usad_arch",
    "score_usad_series",
    "train_usad_source",
    "validate_usad_on_series",
    "DagmmArchConfig",
    "DagmmModel",
    "get_fixed_paper_dagmm_arch",
    "sample_dagmm_arch",
    "score_dagmm_series",
    "train_dagmm_source",
    "validate_dagmm_on_series",
    "TranADArchConfig",
    "TranADModel",
    "get_fixed_paper_tranad_arch",
    "sample_tranad_arch",
    "score_tranad_series",
    "train_tranad_source",
    "validate_tranad_on_series",
]

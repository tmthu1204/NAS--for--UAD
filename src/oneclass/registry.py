from __future__ import annotations

from .autoencoder_backend import AutoEncoderBackend
from .base import OneClassConfig
from .dagmm_head_backend import DAGMMHeadBackend
from .deepsvdd_backend import DeepSVDDBackend
from .drocc_head_backend import DROCCHeadBackend
from .gmm_head_backend import GMMHeadBackend
from .isolation_forest_backend import IsolationForestBackend
from .knn_backend import KNNDistanceBackend
from .lof_head_backend import LOFHeadBackend
from .mahalanobis_head_backend import MahalanobisHeadBackend
from .normalizing_flow_head_backend import NormalizingFlowHeadBackend
from .oneclass_svm_backend import OneClassSVMBackend
from .prototype_oneclass_backend import PrototypeOneClassBackend
from .svdd_backend import SVDDBackend


BACKEND_REGISTRY = {
    DeepSVDDBackend.method_name: DeepSVDDBackend,
    AutoEncoderBackend.method_name: AutoEncoderBackend,
    DAGMMHeadBackend.method_name: DAGMMHeadBackend,
    DROCCHeadBackend.method_name: DROCCHeadBackend,
    KNNDistanceBackend.method_name: KNNDistanceBackend,
    GMMHeadBackend.method_name: GMMHeadBackend,
    IsolationForestBackend.method_name: IsolationForestBackend,
    MahalanobisHeadBackend.method_name: MahalanobisHeadBackend,
    NormalizingFlowHeadBackend.method_name: NormalizingFlowHeadBackend,
    LOFHeadBackend.method_name: LOFHeadBackend,
    OneClassSVMBackend.method_name: OneClassSVMBackend,
    PrototypeOneClassBackend.method_name: PrototypeOneClassBackend,
    SVDDBackend.method_name: SVDDBackend,
}


def list_oneclass_methods():
    return tuple(sorted(BACKEND_REGISTRY))


def get_oneclass_backend_class(method: str):
    key = str(method).strip().lower()
    if key not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown one-class method: {method}. "
            f"Available: {', '.join(list_oneclass_methods())}"
        )
    return BACKEND_REGISTRY[key]


def get_oneclass_score_name(method: str) -> str:
    return get_oneclass_backend_class(method).score_name


def build_oneclass_backend(in_dim: int, config: OneClassConfig):
    backend_cls = get_oneclass_backend_class(config.method)
    return backend_cls(in_dim=in_dim, config=config)

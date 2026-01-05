from dataclasses import dataclass
from typing import List, Tuple, Optional
import random


@dataclass
class ArchConfig:
    # Encoder CNN
    enc_filters: List[int]
    enc_kernels: List[int]
    enc_strides: List[int]
    enc_dilations: List[int]
    enc_pool: Optional[Tuple[str,int]]
    enc_activation: str

    # Sequence model (Transformer / GRU / TCN)
    seq_type: str           # "transformer", "gru", "tcn"
    seq_layers: int
    seq_heads: int          # only for transformer
    seq_hidden: int         # transformer hidden or GRU hidden
    seq_kernel: int         # only for TCN
    seq_dilation: int       # only for TCN

    # Classifier MLP
    clf_layers: int
    clf_units: int

    # Latent dim
    d_model: int


def sample_arch():
    n_enc = random.randint(1, 3)  # 1–3 CNN layers

    enc_filters   = [random.choice([16, 32, 64]) for _ in range(n_enc)]
    enc_kernels   = [random.choice([3, 5, 7])    for _ in range(n_enc)]
    enc_strides   = [random.choice([1, 2])       for _ in range(n_enc)]
    enc_dilations = [random.choice([1, 2])       for _ in range(n_enc)]

    enc_pool = random.choice([None, ('max',2), ('avg',2)])
    enc_act  = random.choice(['relu', 'lrelu'])

    # Sequence model
    seq_type = random.choice(["transformer", "gru", "tcn"])

    if seq_type == "transformer":
        seq_layers = random.randint(1, 2)
        seq_heads  = random.choice([2, 4])
        seq_hidden = random.choice([64, 128])
        seq_kernel, seq_dilation = 3, 1

    elif seq_type == "gru":
        seq_layers = random.randint(1, 2)
        seq_heads  = 1     # ignore
        seq_hidden = random.choice([64, 128])
        seq_kernel, seq_dilation = 3, 1

    else:  # TCN
        seq_layers = random.randint(1, 2)
        seq_heads  = 1     # ignore
        seq_hidden = random.choice([64, 128])
        seq_kernel = random.choice([3, 5])
        seq_dilation = random.choice([1, 2, 4])

    clf_layers = random.randint(1, 3)
    clf_units  = random.choice([32, 64, 128])

    d_model = random.choice([64, 128])

    return ArchConfig(
        enc_filters, enc_kernels, enc_strides, enc_dilations,
        enc_pool, enc_act,
        seq_type, seq_layers, seq_heads, seq_hidden,
        seq_kernel, seq_dilation,
        clf_layers, clf_units, d_model
    )

# ---- Base architectures (hand-designed) for paper baselines ----

def get_base_arches():
    """
    Return a dict of fixed architectures for baselines.
    All are valid ArchConfig in the same search space.
    """
    bases = {}

    # 1) Small CNN + GRU (lightweight baseline)
    bases["CNN_GRU_S"] = ArchConfig(
        enc_filters=[32, 64],
        enc_kernels=[5, 5],
        enc_strides=[1, 2],
        enc_dilations=[1, 1],
        enc_pool=('max', 2),
        enc_activation='relu',
        seq_type="gru",
        seq_layers=1,
        seq_heads=1,
        seq_hidden=64,
        seq_kernel=3,
        seq_dilation=1,
        clf_layers=2,
        clf_units=64,
        d_model=64,
    )

    # 2) Medium CNN + GRU
    bases["CNN_GRU_M"] = ArchConfig(
        enc_filters=[32, 64, 64],
        enc_kernels=[5, 5, 3],
        enc_strides=[1, 2, 1],
        enc_dilations=[1, 1, 1],
        enc_pool=('max', 2),
        enc_activation='relu',
        seq_type="gru",
        seq_layers=2,
        seq_heads=1,
        seq_hidden=128,
        seq_kernel=3,
        seq_dilation=1,
        clf_layers=2,
        clf_units=128,
        d_model=128,
    )

    # 3) Small CNN + Transformer
    bases["CNN_TRF_S"] = ArchConfig(
        enc_filters=[32, 64],
        enc_kernels=[5, 3],
        enc_strides=[1, 2],
        enc_dilations=[1, 1],
        enc_pool=('avg', 2),
        enc_activation='relu',
        seq_type="transformer",
        seq_layers=1,
        seq_heads=2,
        seq_hidden=64,
        seq_kernel=3,
        seq_dilation=1,
        clf_layers=2,
        clf_units=64,
        d_model=64,
    )

    # 4) Medium CNN + Transformer
    bases["CNN_TRF_M"] = ArchConfig(
        enc_filters=[32, 64, 64],
        enc_kernels=[7, 5, 3],
        enc_strides=[1, 2, 1],
        enc_dilations=[1, 1, 1],
        enc_pool=('avg', 2),
        enc_activation='relu',
        seq_type="transformer",
        seq_layers=2,
        seq_heads=4,
        seq_hidden=128,
        seq_kernel=3,
        seq_dilation=1,
        clf_layers=2,
        clf_units=128,
        d_model=128,
    )

    # 5) Small CNN + TCN
    bases["CNN_TCN_S"] = ArchConfig(
        enc_filters=[32, 64],
        enc_kernels=[5, 3],
        enc_strides=[1, 2],
        enc_dilations=[1, 1],
        enc_pool=('max', 2),
        enc_activation='relu',
        seq_type="tcn",
        seq_layers=2,
        seq_heads=1,
        seq_hidden=64,
        seq_kernel=3,
        seq_dilation=2,
        clf_layers=2,
        clf_units=64,
        d_model=64,
    )

    # 6) Medium CNN + TCN (more capacity)
    bases["CNN_TCN_M"] = ArchConfig(
        enc_filters=[32, 64, 64],
        enc_kernels=[7, 5, 3],
        enc_strides=[1, 2, 1],
        enc_dilations=[1, 1, 1],
        enc_pool=('max', 2),
        enc_activation='relu',
        seq_type="tcn",
        seq_layers=2,
        seq_heads=1,
        seq_hidden=128,
        seq_kernel=5,
        seq_dilation=4,
        clf_layers=3,
        clf_units=128,
        d_model=128,
    )

    return bases

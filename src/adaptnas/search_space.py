# from dataclasses import dataclass
# from typing import List, Tuple
# import random


# @dataclass
# class ArchConfig:
#     enc_filters: List[int]
#     enc_kernels: List[int]
#     enc_strides: List[int]
#     enc_pool: Tuple[str,int]
#     enc_activation: str
#     ar_layers: int
#     ar_heads: int
#     ar_hidden: int
#     clf_layers: int
#     clf_units: int
#     d_model: int




# def sample_arch():
#     n = random.randint(1,3)
#     filters = [random.choice([16,32,64]) for _ in range(n)]
#     kernels = [random.choice([3,5]) for _ in range(n)]
#     strides = [random.choice([1,2]) for _ in range(n)]
#     pool = random.choice([('max',2),('avg',2), None])
#     act = random.choice(['relu','lrelu'])
#     ar_l = random.randint(1,3)
#     ar_h = random.choice([2,4])
#     ar_hidden = random.choice([64,128])
#     clf_l = random.randint(1,2)
#     clf_u = random.choice([32,64])
#     d_model = random.choice([64,128])
#     return ArchConfig(filters, kernels, strides, pool, act, ar_l, ar_h, ar_hidden, clf_l, clf_u, d_model)

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

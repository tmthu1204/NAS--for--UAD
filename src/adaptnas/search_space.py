from dataclasses import dataclass
from typing import List, Tuple, Optional
import random


SEARCH_FAMILIES = ("transformer", "gru", "tcn")
ENC_FILTER_CHOICES = (16, 32, 64)
ENC_KERNEL_CHOICES = (3, 5, 7)
ENC_STRIDE_CHOICES = (1, 2)
ENC_DILATION_CHOICES = (1, 2)
ENC_POOL_CHOICES = (None, ("max", 2), ("avg", 2))
ENC_ACT_CHOICES = ("relu", "lrelu")
SEQ_LAYER_CHOICES = (1, 2)
SEQ_HEAD_CHOICES = (2, 4)
SEQ_HIDDEN_CHOICES = (64, 128)
SEQ_KERNEL_CHOICES = (3, 5)
SEQ_DILATION_CHOICES = (1, 2, 4)
CLF_LAYER_CHOICES = (1, 2, 3)
CLF_UNIT_CHOICES = (32, 64, 128)
D_MODEL_CHOICES = (64, 128)


def _search_space_choices(compact: bool = False):
    if not compact:
        return {
            "n_enc_choices": (1, 2, 3),
            "enc_filters": ENC_FILTER_CHOICES,
            "enc_kernels": ENC_KERNEL_CHOICES,
            "enc_strides": ENC_STRIDE_CHOICES,
            "enc_dilations": ENC_DILATION_CHOICES,
            "enc_pool": ENC_POOL_CHOICES,
            "enc_activation": ENC_ACT_CHOICES,
            "seq_layers": SEQ_LAYER_CHOICES,
            "seq_heads": SEQ_HEAD_CHOICES,
            "seq_hidden": SEQ_HIDDEN_CHOICES,
            "seq_kernel": SEQ_KERNEL_CHOICES,
            "seq_dilation": SEQ_DILATION_CHOICES,
            "clf_layers": CLF_LAYER_CHOICES,
            "clf_units": CLF_UNIT_CHOICES,
            "d_model": D_MODEL_CHOICES,
        }
    return {
        "n_enc_choices": (1, 2),
        "enc_filters": (32, 64),
        "enc_kernels": (3, 5),
        "enc_strides": (1, 2),
        "enc_dilations": (1,),
        "enc_pool": (None, ("max", 2)),
        "enc_activation": ("relu",),
        "seq_layers": SEQ_LAYER_CHOICES,
        "seq_heads": SEQ_HEAD_CHOICES,
        "seq_hidden": SEQ_HIDDEN_CHOICES,
        "seq_kernel": SEQ_KERNEL_CHOICES,
        "seq_dilation": (1, 2),
        "clf_layers": (1, 2),
        "clf_units": (64, 128),
        "d_model": D_MODEL_CHOICES,
    }


@dataclass
class ArchConfig:
    # Encoder CNN
    enc_filters: List[int]
    enc_kernels: List[int]
    enc_strides: List[int]
    enc_dilations: List[int]
    enc_pool: Optional[Tuple[str, int]]
    enc_activation: str

    # Sequence model (Transformer / GRU / TCN)
    seq_type: str
    seq_layers: int
    seq_heads: int
    seq_hidden: int
    seq_kernel: int
    seq_dilation: int

    # Classifier MLP
    clf_layers: int
    clf_units: int

    # Latent dim
    d_model: int


def list_search_families():
    return list(SEARCH_FAMILIES)


def _choose_alt(current, choices):
    alternatives = [choice for choice in choices if choice != current]
    if not alternatives:
        return current
    return random.choice(alternatives)


def clone_arch(arch: "ArchConfig"):
    return ArchConfig(
        enc_filters=list(arch.enc_filters),
        enc_kernels=list(arch.enc_kernels),
        enc_strides=list(arch.enc_strides),
        enc_dilations=list(arch.enc_dilations),
        enc_pool=None if arch.enc_pool is None else (arch.enc_pool[0], arch.enc_pool[1]),
        enc_activation=str(arch.enc_activation),
        seq_type=str(arch.seq_type),
        seq_layers=int(arch.seq_layers),
        seq_heads=int(arch.seq_heads),
        seq_hidden=int(arch.seq_hidden),
        seq_kernel=int(arch.seq_kernel),
        seq_dilation=int(arch.seq_dilation),
        clf_layers=int(arch.clf_layers),
        clf_units=int(arch.clf_units),
        d_model=int(arch.d_model),
    )


def _retarget_seq_family(arch: "ArchConfig", seq_type: str):
    if seq_type not in SEARCH_FAMILIES:
        raise ValueError(f"Unsupported seq_type: {seq_type}. Expected one of {SEARCH_FAMILIES}.")

    arch.seq_type = seq_type
    arch.seq_layers = random.choice(SEQ_LAYER_CHOICES)
    arch.seq_hidden = random.choice(SEQ_HIDDEN_CHOICES)

    if seq_type == "transformer":
        arch.seq_heads = random.choice(SEQ_HEAD_CHOICES)
        arch.seq_kernel = 3
        arch.seq_dilation = 1
    elif seq_type == "gru":
        arch.seq_heads = 1
        arch.seq_kernel = 3
        arch.seq_dilation = 1
    else:
        arch.seq_heads = 1
        arch.seq_kernel = random.choice(SEQ_KERNEL_CHOICES)
        arch.seq_dilation = random.choice(SEQ_DILATION_CHOICES)

    return arch


def sample_arch(seq_type: Optional[str] = None, compact: bool = False):
    choices = _search_space_choices(compact=compact)
    n_enc = random.choice(choices["n_enc_choices"])

    enc_filters = [random.choice(choices["enc_filters"]) for _ in range(n_enc)]
    enc_kernels = [random.choice(choices["enc_kernels"]) for _ in range(n_enc)]
    enc_strides = [random.choice(choices["enc_strides"]) for _ in range(n_enc)]
    enc_dilations = [random.choice(choices["enc_dilations"]) for _ in range(n_enc)]

    enc_pool = random.choice(choices["enc_pool"])
    enc_act = random.choice(choices["enc_activation"])

    if seq_type is None:
        seq_type = random.choice(list_search_families())
    elif seq_type not in SEARCH_FAMILIES:
        raise ValueError(f"Unsupported seq_type: {seq_type}. Expected one of {SEARCH_FAMILIES}.")

    if seq_type == "transformer":
        seq_layers = random.choice(choices["seq_layers"])
        seq_heads = random.choice(choices["seq_heads"])
        seq_hidden = random.choice(choices["seq_hidden"])
        seq_kernel, seq_dilation = 3, 1
    elif seq_type == "gru":
        seq_layers = random.choice(choices["seq_layers"])
        seq_heads = 1
        seq_hidden = random.choice(choices["seq_hidden"])
        seq_kernel, seq_dilation = 3, 1
    else:
        seq_layers = random.choice(choices["seq_layers"])
        seq_heads = 1
        seq_hidden = random.choice(choices["seq_hidden"])
        seq_kernel = random.choice(choices["seq_kernel"])
        seq_dilation = random.choice(choices["seq_dilation"])

    clf_layers = random.choice(choices["clf_layers"])
    clf_units = random.choice(choices["clf_units"])
    d_model = random.choice(choices["d_model"])

    return ArchConfig(
        enc_filters,
        enc_kernels,
        enc_strides,
        enc_dilations,
        enc_pool,
        enc_act,
        seq_type,
        seq_layers,
        seq_heads,
        seq_hidden,
        seq_kernel,
        seq_dilation,
        clf_layers,
        clf_units,
        d_model,
    )


def mutate_arch(
    parent: "ArchConfig",
    *,
    allowed_families: Optional[List[str]] = None,
    mutation_steps: int = 2,
    target_family: Optional[str] = None,
    force_family_change: bool = False,
):
    arch = clone_arch(parent)
    allowed = list(allowed_families) if allowed_families else list_search_families()
    allowed = [family for family in allowed if family in SEARCH_FAMILIES]
    if not allowed:
        allowed = list_search_families()

    if target_family is not None:
        if target_family not in allowed:
            raise ValueError(f"Unsupported target_family: {target_family}. Expected one of {allowed}.")
        if target_family != arch.seq_type:
            arch = _retarget_seq_family(arch, target_family)
    elif force_family_change and len(allowed) > 1:
        arch = _retarget_seq_family(arch, _choose_alt(arch.seq_type, allowed))

    for _ in range(max(1, int(mutation_steps))):
        ops = [
            "enc_depth",
            "enc_filter",
            "enc_kernel",
            "enc_stride",
            "enc_dilation",
            "enc_pool",
            "enc_activation",
            "seq_layers",
            "seq_hidden",
            "clf_layers",
            "clf_units",
            "d_model",
        ]
        if len(allowed) > 1:
            ops.append("seq_type")
        if arch.seq_type == "transformer":
            ops.append("seq_heads")
        elif arch.seq_type == "tcn":
            ops.extend(["seq_kernel", "seq_dilation"])

        op = random.choice(ops)

        if op == "enc_depth":
            if len(arch.enc_filters) == 1:
                action = "add"
            elif len(arch.enc_filters) == 3:
                action = "remove"
            else:
                action = random.choice(["add", "remove"])
            if action == "add":
                arch.enc_filters.append(random.choice(ENC_FILTER_CHOICES))
                arch.enc_kernels.append(random.choice(ENC_KERNEL_CHOICES))
                arch.enc_strides.append(random.choice(ENC_STRIDE_CHOICES))
                arch.enc_dilations.append(random.choice(ENC_DILATION_CHOICES))
            else:
                remove_idx = random.randrange(len(arch.enc_filters))
                arch.enc_filters.pop(remove_idx)
                arch.enc_kernels.pop(remove_idx)
                arch.enc_strides.pop(remove_idx)
                arch.enc_dilations.pop(remove_idx)
        elif op == "enc_filter":
            idx = random.randrange(len(arch.enc_filters))
            arch.enc_filters[idx] = _choose_alt(arch.enc_filters[idx], ENC_FILTER_CHOICES)
        elif op == "enc_kernel":
            idx = random.randrange(len(arch.enc_kernels))
            arch.enc_kernels[idx] = _choose_alt(arch.enc_kernels[idx], ENC_KERNEL_CHOICES)
        elif op == "enc_stride":
            idx = random.randrange(len(arch.enc_strides))
            arch.enc_strides[idx] = _choose_alt(arch.enc_strides[idx], ENC_STRIDE_CHOICES)
        elif op == "enc_dilation":
            idx = random.randrange(len(arch.enc_dilations))
            arch.enc_dilations[idx] = _choose_alt(arch.enc_dilations[idx], ENC_DILATION_CHOICES)
        elif op == "enc_pool":
            arch.enc_pool = _choose_alt(arch.enc_pool, ENC_POOL_CHOICES)
        elif op == "enc_activation":
            arch.enc_activation = _choose_alt(arch.enc_activation, ENC_ACT_CHOICES)
        elif op == "seq_type":
            new_family = _choose_alt(arch.seq_type, allowed)
            arch = _retarget_seq_family(arch, new_family)
        elif op == "seq_layers":
            arch.seq_layers = _choose_alt(arch.seq_layers, SEQ_LAYER_CHOICES)
        elif op == "seq_hidden":
            arch.seq_hidden = _choose_alt(arch.seq_hidden, SEQ_HIDDEN_CHOICES)
        elif op == "seq_heads" and arch.seq_type == "transformer":
            arch.seq_heads = _choose_alt(arch.seq_heads, SEQ_HEAD_CHOICES)
        elif op == "seq_kernel" and arch.seq_type == "tcn":
            arch.seq_kernel = _choose_alt(arch.seq_kernel, SEQ_KERNEL_CHOICES)
        elif op == "seq_dilation" and arch.seq_type == "tcn":
            arch.seq_dilation = _choose_alt(arch.seq_dilation, SEQ_DILATION_CHOICES)
        elif op == "clf_layers":
            arch.clf_layers = _choose_alt(arch.clf_layers, CLF_LAYER_CHOICES)
        elif op == "clf_units":
            arch.clf_units = _choose_alt(arch.clf_units, CLF_UNIT_CHOICES)
        elif op == "d_model":
            arch.d_model = _choose_alt(arch.d_model, D_MODEL_CHOICES)

    if target_family is not None and arch.seq_type != target_family:
        arch = _retarget_seq_family(arch, target_family)
    elif force_family_change and len(allowed) > 1 and arch.seq_type == parent.seq_type:
        arch = _retarget_seq_family(arch, _choose_alt(parent.seq_type, allowed))

    return arch


def get_base_arches():
    """
    Return a dict of fixed architectures for baselines.
    All are valid ArchConfig in the same search space.
    """
    bases = {}

    bases["CNN_GRU_S"] = ArchConfig(
        enc_filters=[32, 64],
        enc_kernels=[5, 5],
        enc_strides=[1, 2],
        enc_dilations=[1, 1],
        enc_pool=("max", 2),
        enc_activation="relu",
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

    bases["CNN_GRU_M"] = ArchConfig(
        enc_filters=[32, 64, 64],
        enc_kernels=[5, 5, 3],
        enc_strides=[1, 2, 1],
        enc_dilations=[1, 1, 1],
        enc_pool=("max", 2),
        enc_activation="relu",
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

    bases["CNN_TRF_S"] = ArchConfig(
        enc_filters=[32, 64],
        enc_kernels=[5, 3],
        enc_strides=[1, 2],
        enc_dilations=[1, 1],
        enc_pool=("avg", 2),
        enc_activation="relu",
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

    bases["CNN_TRF_M"] = ArchConfig(
        enc_filters=[32, 64, 64],
        enc_kernels=[7, 5, 3],
        enc_strides=[1, 2, 1],
        enc_dilations=[1, 1, 1],
        enc_pool=("avg", 2),
        enc_activation="relu",
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

    bases["CNN_TCN_S"] = ArchConfig(
        enc_filters=[32, 64],
        enc_kernels=[5, 3],
        enc_strides=[1, 2],
        enc_dilations=[1, 1],
        enc_pool=("max", 2),
        enc_activation="relu",
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

    bases["CNN_TCN_M"] = ArchConfig(
        enc_filters=[32, 64, 64],
        enc_kernels=[7, 5, 3],
        enc_strides=[1, 2, 1],
        enc_dilations=[1, 1, 1],
        enc_pool=("max", 2),
        enc_activation="relu",
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

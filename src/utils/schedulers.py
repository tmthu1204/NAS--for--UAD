import math


def exp_grl(p: float) -> float:
    # Ganin et al. schedule
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


def cosine_grl(p: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * p))
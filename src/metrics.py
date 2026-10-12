import numpy as np
from scipy.stats import wilcoxon


def masked_errors(pred: np.ndarray, true: np.ndarray,
                  missing: np.ndarray):
    m = missing == 1
    sq = ((pred - true) ** 2)[m]
    ab = np.abs(pred - true)[m]
    return sq, ab


def crps_empirical(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    S = samples.shape[0]
    t1 = np.abs(samples - y[None]).mean(0)
    if S < 2:
        return np.zeros_like(t1)
    ss = np.sort(samples, axis=0)
    i = np.arange(S).reshape((S,) + (1,) * y.ndim)
    w = (2.0 * i - S + 1.0)
    t2 = (w * ss).sum(0) * (2.0 / (S * S))
    return t1 - 0.5 * t2


def interval_coverage(samples: np.ndarray, y: np.ndarray,
                      level: float = 0.9) -> np.ndarray:
    lo = np.quantile(samples, (1.0 - level) / 2, axis=0)
    hi = np.quantile(samples, 1.0 - (1.0 - level) / 2, axis=0)
    return ((y >= lo) & (y <= hi)).astype(np.float32)


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 5 or np.allclose(a, b):
        return float("nan"), float("nan")
    try:
        stat, p = wilcoxon(a, b)
        return float(stat), float(p)
    except ValueError:
        return float("nan"), float("nan")


def fmt_p(p) -> str:
    if p != p:
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

"""CFAR detectors (floating point) on square-law (power) data.

CA-CFAR with N training cells (total, both sides), exponential noise:
    threshold T = (alpha / N) * sum(training),  alpha = N * (Pfa^(-1/N) - 1)
gives exactly the design Pfa for i.i.d. exponential noise.
OS-CFAR: T = alpha_os * (k-th smallest training cell); alpha_os solved numerically from
    Pfa = prod_{i=0}^{k-1} (N - i) / (N - i + alpha_os)   (Rohling 1983).

1D edge handling (MUST match the hardware): only cells with a complete window on both
sides are tested; the first and last (n_train + n_guard) cells of a line are never
declared detections.
2D CA-CFAR: rectangular window of training cells around a guard rectangle; edge cells
without a complete window are not tested; Doppler wraps around (circular), range doesn't.
"""
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.optimize import brentq


def ca_alpha(n, pfa):
    return n * (pfa ** (-1.0 / n) - 1.0)


def os_pfa(alpha, n, k):
    i = np.arange(k)
    return float(np.prod((n - i) / (n - i + alpha)))


def os_alpha(n, k, pfa):
    return brentq(lambda a: os_pfa(a, n, k) - pfa, 1e-6, 1e6)


def ca_pd_swerling1(snr_lin, n, pfa):
    """Closed form Pd for a Swerling-1 target in CA-CFAR (i.i.d. exponential noise)."""
    a = ca_alpha(n, pfa)
    return (1.0 + a / (n * (1.0 + snr_lin))) ** (-n)


def _windows_1d(x, n_train, n_guard):
    """Leading and lagging training sums for every cell with a full window."""
    x = np.asarray(x, np.float64)
    L = x.shape[-1]
    h = n_train + n_guard
    c = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, axis=-1)], axis=-1)
    idx = np.arange(h, L - h)
    lag = c[..., idx - n_guard] - c[..., idx - h]               # cells idx-h .. idx-n_guard-1
    lead = c[..., idx + h + 1] - c[..., idx + n_guard + 1]      # cells idx+n_guard+1 .. idx+h
    return idx, lag, lead


def ca_cfar_1d(x, n_train, n_guard, pfa=None, alpha=None):
    """Returns boolean detections (same shape as x) and the threshold (NaN at edges)."""
    x = np.asarray(x, np.float64)
    n = 2 * n_train
    a = ca_alpha(n, pfa) if alpha is None else alpha
    idx, lag, lead = _windows_1d(x, n_train, n_guard)
    thr = np.full(x.shape, np.nan)
    thr[..., idx] = a / n * (lag + lead)
    det = np.zeros(x.shape, bool)
    det[..., idx] = x[..., idx] > thr[..., idx]
    return det, thr


def os_cfar_1d(x, n_train, n_guard, pfa, k=None):
    x = np.asarray(x, np.float64)
    n = 2 * n_train
    k = k or int(round(0.75 * n))
    a = os_alpha(n, k, pfa)
    L = x.shape[-1]
    h = n_train + n_guard
    det = np.zeros(x.shape, bool)
    thr = np.full(x.shape, np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    win = sliding_window_view(x, 2 * h + 1, axis=-1)            # (..., L-2h, 2h+1)
    train = np.concatenate([win[..., :n_train], win[..., n_train + 2 * n_guard + 1:]], axis=-1)
    kth = np.partition(train, k - 1, axis=-1)[..., k - 1]
    thr[..., h:L - h] = a * kth
    det[..., h:L - h] = x[..., h:L - h] > thr[..., h:L - h]
    return det, thr


def ca_cfar_2d(p, train=(8, 4), guard=(2, 2), pfa=1e-4):
    """p: (Nd, Nr) power map. train/guard = (range, doppler) half-widths."""
    tr, td = train
    gr, gd = guard
    outer = (2 * (td + gd) + 1, 2 * (tr + gr) + 1)
    inner = (2 * gd + 1, 2 * gr + 1)
    n = outer[0] * outer[1] - inner[0] * inner[1]
    a = ca_alpha(n, pfa)
    s_out = uniform_filter(p, size=outer, mode=("wrap", "constant")) * outer[0] * outer[1]
    s_in = uniform_filter(p, size=inner, mode=("wrap", "constant")) * inner[0] * inner[1]
    thr = a / n * (s_out - s_in)
    det = p > thr
    h = tr + gr
    det[:, :h] = False
    det[:, -h:] = False
    return det, thr, n


def cluster(det, p):
    """Connected components (8-connectivity) -> reports with the peak cell of each."""
    from scipy.ndimage import label, maximum_position
    lab, k = label(det, structure=np.ones((3, 3)))
    if k == 0:
        return []
    peaks = maximum_position(p, lab, index=np.arange(1, k + 1))
    return [{"doppler_bin": int(d), "range_bin": int(r), "power": float(p[d, r]),
             "cells": int((lab == i + 1).sum())} for i, (d, r) in enumerate(peaks)]

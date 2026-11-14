"""Bit-exact fixed-point model of rtl/cfar_ca_1d.v (1D CA-CFAR along range).

Formats
- input cells: unsigned DATA_W-bit integers (default 16): quantized power, see quantize_power
- alpha_q: unsigned ALPHA_W-bit (default 17) with ALPHA_FRAC = 12 fractional bits (Q5.12,
  0 <= alpha < 32). alpha_q = round(alpha * 2^12). 17 bits keep the product in one DSP48.
- training sums: exact integers (lead side + lag side), N = 2 * n_train cells
- decision (no division, no rounding):  det = cut * N * 2^ALPHA_FRAC  >  alpha_q * (lead + lag)
  which is exactly  cut > (alpha_q / 2^12) / N * sum.

Window and edge rule (identical to the RTL and to src/proc/cfar.ca_cfar_1d): for cell
index k of a line, the RTL emits a decision for CUT = k - (n_train + n_guard) once
k >= W - 1, W = 2 (n_train + n_guard) + 1. Cells within n_train + n_guard of either end of a
line get no decision (never detected). Outputs per line: (cut_index, det) in index order.
"""
import numpy as np

ALPHA_FRAC = 12
ALPHA_W = 17
DATA_W = 16


def quantize_alpha(alpha, frac=ALPHA_FRAC, width=ALPHA_W):
    q = int(round(alpha * (1 << frac)))
    if not 0 <= q < (1 << width):
        raise ValueError(f"alpha {alpha} does not fit Q{width - frac}.{frac}")
    return q


def quantize_power(p, noise_mean_code=64, data_w=DATA_W):
    """Power map in noise units (unit mean noise) -> unsigned codes: floor(p * 64), saturate.
    64 codes per unit noise power leaves 30.1 dB of headroom above the noise mean."""
    q = np.floor(np.asarray(p, np.float64) * noise_mean_code)
    return np.clip(q, 0, (1 << data_w) - 1).astype(np.int64)


def cfar_line(x, n_train, n_guard, alpha_q, frac=ALPHA_FRAC):
    """x: 1D integer array (one range line). Returns (cut_idx, det, sum, cut) arrays for the
    cells that receive a decision."""
    x = np.asarray(x, np.int64)
    L = x.size
    h = n_train + n_guard
    if L < 2 * h + 1:
        e = np.zeros(0, np.int64)
        return e, e.astype(bool), e, e
    c = np.concatenate([[0], np.cumsum(x)])
    idx = np.arange(h, L - h)
    lag = c[idx - n_guard] - c[idx - h]
    lead = c[idx + h + 1] - c[idx + n_guard + 1]
    s = lag + lead
    n = 2 * n_train
    lhs = x[idx] * n * (1 << frac)
    rhs = alpha_q * s
    return idx, lhs > rhs, s, x[idx]


def cfar_lines(lines, n_train, n_guard, alpha_q):
    return [cfar_line(l, n_train, n_guard, alpha_q) for l in lines]


def widths(n_train, data_w=DATA_W, alpha_w=ALPHA_W, frac=ALPHA_FRAC):
    """Bit widths needed (see docs/EXPLAINER.md overflow analysis)."""
    n = 2 * n_train
    side = data_w + int(np.ceil(np.log2(n_train + 1)))
    total = data_w + int(np.ceil(np.log2(n + 1)))
    lhs = data_w + int(np.ceil(np.log2(n + 1))) + frac
    rhs = alpha_w + total
    return {"side_sum": side, "total_sum": total, "lhs": lhs, "rhs": rhs, "cmp": max(lhs, rhs)}

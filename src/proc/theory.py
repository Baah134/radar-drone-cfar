"""Closed-form / numerical CFAR detection theory (square-law, unit-mean exponential noise).

Swerling 1 (scan-to-scan Rayleigh), CA-CFAR with N training cells:
    Pd = (1 + alpha / (N (1 + SNR)))^(-N)
Swerling 0 (constant amplitude), CA-CFAR: no simple closed form. With training sum
S ~ Gamma(N, 1) and threshold T = alpha S / N, the CUT power |s + n|^2 satisfies
2|s+n|^2 ~ noncentral chi^2 (2 dof, nc = 2 SNR), so
    Pd = E_S[ ncx2.sf(2 alpha S / N; 2, 2 SNR) ]
evaluated by Gauss-Laguerre-free numerical integration over the Gamma density.
"""
import numpy as np
from scipy import integrate, stats

from .cfar import ca_alpha, ca_pd_swerling1


def ca_pd_swerling0(snr_lin, n, pfa):
    a = ca_alpha(n, pfa)
    g = stats.gamma(n)
    lo, hi = g.ppf(1e-12), g.ppf(1 - 1e-12)
    f = lambda s: stats.ncx2.sf(2 * a * s / n, 2, 2 * snr_lin) * g.pdf(s)
    return integrate.quad(f, lo, hi, limit=200, epsabs=1e-10)[0]


__all__ = ["ca_pd_swerling0", "ca_pd_swerling1"]


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def clustered_ci(counts, sizes, z=1.96):
    """CI for a rate when trials come in correlated clusters (e.g. cells of one CFAR line,
    which share training cells). Ratio estimator with cluster-level variance (normal approx.):
    rate = sum(counts)/sum(sizes); se from the per-cluster residuals."""
    c = np.asarray(counts, float)
    m = np.asarray(sizes, float)
    k = len(c)
    r = c.sum() / m.sum()
    se = np.sqrt(np.sum((c - r * m) ** 2) / (k * (k - 1))) / m.mean()
    return (max(0.0, r - z * se), r + z * se)

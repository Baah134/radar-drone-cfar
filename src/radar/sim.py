"""FMCW scene simulation: dechirped complex IF signal per chirp.

A point scatterer at range R(t) contributes, for fast-time sample n and chirp m,
    a * exp(j*2*pi*(f_b * n / fs)) * exp(j*4*pi*R(t_m)/lam)
with beat frequency f_b = 2*S*R/c. R(t_m) is evaluated per chirp (slow time), so the
chirp-to-chirp phase carries the Doppler. Range migration within a frame is ignored:
at 30 m/s over a 12.8 ms frame a target moves 0.38 m < 0.6 m range bin (stated in docs).
The intra-chirp Doppler shift (2v/lam, at most ~5 kHz for 31 m/s, vs the 10 kHz
fast-time bin spacing) is under one range bin and is ignored.

Amplitudes are set from a target SNR so detection experiments can be specified directly:
noise is complex white Gaussian with unit variance per sample.
"""
import numpy as np

from .params import FMCW, C


def beat_signal(scatterers, radar: FMCW, rng, noise=True, clutter=None):
    """scatterers: list of dicts with keys
         r0 (m), v (m/s), amp (complex or callable(m) -> complex per chirp),
         or 'r_of_t': callable(t array) -> range array (m) for arbitrary motion.
    Returns complex array (Nc, Ns)."""
    Nc, Ns = radar.Nc, radar.Ns
    n = np.arange(Ns)
    t = np.arange(Nc) * radar.Tc
    x = np.zeros((Nc, Ns), np.complex128)
    for s in scatterers:
        R = s["r_of_t"](t) if "r_of_t" in s else s["r0"] + s.get("v", 0.0) * t
        a = s["amp"](np.arange(Nc)) if callable(s["amp"]) else np.full(Nc, s["amp"], complex)
        fb = 2 * radar.slope * R / C                       # (Nc,)
        ph = 4 * np.pi * R / radar.lam                     # (Nc,)
        x += (a * np.exp(1j * ph))[:, None] * np.exp(1j * 2 * np.pi * fb[:, None] * n[None, :] / radar.fs)
    if clutter is not None:
        x += clutter
    if noise:
        x += (rng.standard_normal((Nc, Ns)) + 1j * rng.standard_normal((Nc, Ns))) / np.sqrt(2)
    return x


def ground_clutter(radar: FMCW, rng, cnr_db=30.0, r_min=2.0, r_max=None, spread_mps=0.1, k_shape=None,
                   edge=None):
    """Stationary ground clutter: one scatterer per range bin with Rayleigh (or K-distributed
    if k_shape is given) amplitude, small random Doppler spread (wind, antenna jitter).
    cnr_db: clutter-to-noise ratio per sample. edge: (r_edge, extra_db) adds a clutter
    step at r_edge (clutter-edge scene)."""
    r_max = r_max or radar.range_max * 0.95
    bins = np.arange(int(r_min / radar.range_res), int(r_max / radar.range_res))
    scat = []
    p = 10 ** (cnr_db / 10)
    for b in bins:
        r = (b + rng.uniform(-0.3, 0.3)) * radar.range_res
        pw = p
        if edge is not None and r >= edge[0]:
            pw = p * 10 ** (edge[1] / 10)
        if k_shape is not None:
            pw *= rng.gamma(k_shape, 1 / k_shape)          # K-distribution: gamma texture x speckle
        amp0 = np.sqrt(pw) * (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
        fd = rng.normal(0, 2 * spread_mps / radar.lam)     # Doppler spread (Hz)
        scat.append({"r0": r, "v": 0.0,
                     "amp": (lambda m, a=amp0, f=fd: a * np.exp(1j * 2 * np.pi * f * m * radar.Tc))})
    return scat


def amp_for_snr(snr_db, radar: FMCW, window_gain=1.0):
    """Per-sample amplitude giving a post-FFT (range-Doppler cell) SNR of snr_db, for unit
    noise variance per sample. Coherent gain of an Ns x Nc FFT: signal power scales by
    (Ns*Nc)^2 * cg^2, noise by Ns*Nc * enbw-related factor; with window w the cell SNR is
    |a|^2 * (sum w)^2 / sum w^2 over both dimensions. window_gain = that ratio / (Ns*Nc)."""
    g = radar.Ns * radar.Nc * window_gain
    return np.sqrt(10 ** (snr_db / 10) / g)

"""Processing chain: window -> range FFT -> clutter suppression -> Doppler FFT -> |.|^2.

Windows: Hann (default; -31 dB first sidelobe, 1.5-bin noise bandwidth) or Blackman
(-58 dB sidelobes, wider mainlobe). Clutter suppression along slow time per range bin:
  none, mean (subtract the slow-time mean: removes exactly-zero Doppler),
  mti2 (two-pulse canceller y[m] = x[m] - x[m-1]), mti3 (three-pulse x[m] - 2x[m-1] + x[m-2]).
The output map is normalized so that unit-variance white noise gives unit mean power per
cell (so CFAR thresholds and SNRs are in noise units).
"""
import numpy as np

WINDOWS = {"hann": np.hanning, "blackman": np.blackman, "rect": np.ones}


def _win(name, n):
    w = WINDOWS[name](n)
    return w.astype(np.float64)


def window_snr_gain(name, n):
    """(sum w)^2 / (n * sum w^2): coherent SNR gain of a window relative to rectangular."""
    w = _win(name, n)
    return w.sum() ** 2 / (n * (w ** 2).sum())


def range_fft(x, window="hann"):
    Nc, Ns = x.shape
    w = _win(window, Ns)
    return np.fft.fft(x * w[None, :], axis=1) / np.sqrt((w ** 2).sum())


def suppress(rng_prof, method="none"):
    """rng_prof: (Nc, Ns) after range FFT. Returns slow-time-filtered profiles and the
    noise power gain of the filter (to keep noise normalized to unit power)."""
    if method == "none":
        return rng_prof, 1.0
    if method == "mean":
        Nc = rng_prof.shape[0]
        return rng_prof - rng_prof.mean(axis=0, keepdims=True), (Nc - 1) / Nc
    if method == "mti2":
        return rng_prof[1:] - rng_prof[:-1], 2.0
    if method == "mti3":
        return rng_prof[2:] - 2 * rng_prof[1:-1] + rng_prof[:-2], 6.0
    raise ValueError(method)


def doppler_fft(rp, window="hann", nfft=None):
    Nc = rp.shape[0]
    nfft = nfft or Nc
    w = _win(window, Nc)
    y = np.fft.fft(rp * w[:, None], n=nfft, axis=0) / np.sqrt((w ** 2).sum())
    return np.fft.fftshift(y, axes=0)


def rd_map(x, window="hann", clutter="none", nfft_doppler=None):
    """Complex IF (Nc, Ns) -> power map (Ndoppler, Ns), unit mean for unit-variance noise.
    Doppler axis is fftshifted (zero velocity at index Nd//2)."""
    rp = range_fft(x, window)
    rp, g = suppress(rp, clutter)
    nd = nfft_doppler or x.shape[0]
    y = doppler_fft(rp, window, nfft=nd)
    return (np.abs(y) ** 2) / g


def bin_to_range(b, radar):
    return b * radar.range_res


def bin_to_velocity(d, radar, nd=None):
    nd = nd or radar.Nc
    return (d - nd // 2) * radar.lam / (2 * nd * radar.Tc)

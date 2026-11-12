import numpy as np

from src.proc.chain import bin_to_velocity, rd_map
from src.radar.params import C, FMCW
from src.radar.sim import amp_for_snr, beat_signal


R = FMCW()


def test_derived_parameters():
    assert np.isclose(R.range_res, C / (2 * 250e6))
    assert np.isclose(R.range_res, 0.5996, atol=1e-3)
    assert np.isclose(R.v_max, (C / 24e9) / (4 * 100e-6))
    assert np.isclose(R.v_res, (C / 24e9) / (2 * 128 * 100e-6))
    assert np.isclose(R.range_max, R.Ns * R.range_res)


def _peak(r0, v):
    rng = np.random.default_rng(0)
    x = beat_signal([{"r0": r0, "v": v, "amp": amp_for_snr(30, R)}], R, rng)
    p = rd_map(x, "hann", "none")
    d, b = np.unravel_index(np.argmax(p), p.shape)
    return d, b, p


def test_stationary_target_range_bin():
    for r0 in (10.0, 37.3, 90.1):
        d, b, _ = _peak(r0, 0.0)
        assert abs(b - r0 / R.range_res) <= 1
        assert d == R.Nc // 2


def test_moving_target_doppler_bin():
    for v in (-12.0, 3.3, 20.0):
        d, b, _ = _peak(50.0, v)
        assert abs(bin_to_velocity(d, R) - v) <= R.v_res
        assert abs(b - 50.0 / R.range_res) <= 1


def test_noise_normalization_unit_mean():
    rng = np.random.default_rng(1)
    p = rd_map(beat_signal([], R, rng), "hann", "none")
    assert abs(p.mean() - 1.0) < 0.02


def test_snr_calibration():
    """amp_for_snr with window gain gives the requested peak cell SNR (on-bin target)."""
    from src.proc.chain import window_snr_gain
    g = window_snr_gain("hann", R.Ns) * window_snr_gain("hann", R.Nc)
    x = beat_signal([{"r0": 40 * R.range_res, "v": 0.0, "amp": amp_for_snr(20, R, g)}], R, None, noise=False)
    p = rd_map(x, "hann", "none")
    assert abs(10 * np.log10(p.max()) - 20) < 0.05


def test_determinism():
    a = beat_signal([], R, np.random.default_rng(5))
    b = beat_signal([], R, np.random.default_rng(5))
    assert np.array_equal(a, b)

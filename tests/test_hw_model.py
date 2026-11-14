import numpy as np

from src.hw_model.cfar_fixed import ALPHA_W, cfar_line, quantize_alpha, quantize_power, widths
from src.proc.cfar import ca_alpha, ca_cfar_1d


def test_fixed_matches_float_on_clean_cases():
    """Same window and edge rule; with alpha exactly representable both agree exactly."""
    rng = np.random.default_rng(0)
    nt, ng = 16, 2
    aq = quantize_alpha(ca_alpha(2 * nt, 1e-4))
    a_eff = aq / 4096
    for _ in range(50):
        x = np.minimum(65535, np.floor(rng.exponential(64, 300))).astype(np.int64)
        x[rng.integers(0, 300, 3)] += 20000
        idx, det, _, _ = cfar_line(x, nt, ng, aq)
        fdet, thr = ca_cfar_1d(x.astype(float), nt, ng, alpha=a_eff)
        assert np.array_equal(idx, np.nonzero(np.isfinite(thr))[0])
        # float threshold vs exact integer compare can differ only for exact ties
        tie = np.isclose(x[idx], thr[idx], rtol=1e-12, atol=0)
        assert np.array_equal(det[~tie], fdet[idx][~tie])


def test_edge_rule_and_short_lines():
    nt, ng = 8, 2
    W = 2 * (nt + ng) + 1
    idx, det, _, _ = cfar_line(np.ones(W - 1, np.int64), nt, ng, 1000)
    assert idx.size == 0
    idx, _, _, _ = cfar_line(np.ones(W, np.int64), nt, ng, 1000)
    assert idx.tolist() == [nt + ng]


def test_alpha_quantization_fits_and_dsp_width():
    assert ALPHA_W == 17
    for n in (16, 32, 64):
        for pfa in (1e-4, 1e-5, 1e-6):
            q = quantize_alpha(ca_alpha(n, pfa))
            assert 0 < q < (1 << 17)
            assert abs(q / 4096 - ca_alpha(n, pfa)) <= 0.5 / 4096


def test_overflow_bounds():
    """Worst case: every cell at full scale. Products must fit the declared widths."""
    for nt in (8, 16, 32, 64):
        w = widths(nt)
        n = 2 * nt
        assert nt * 65535 < (1 << w["side_sum"])
        assert n * 65535 < (1 << w["total_sum"])
        assert 65535 * n * 4096 < (1 << w["lhs"])
        assert ((1 << 17) - 1) * n * 65535 < (1 << w["rhs"])
        assert w["total_sum"] <= 24          # fits the DSP48 25-bit signed A port as unsigned


def test_quantize_power_saturates():
    q = quantize_power(np.array([0.0, 1.0, 1e9]))
    assert q.tolist() == [0, 64, 65535]

import numpy as np

from src.proc.cfar import ca_alpha, ca_cfar_1d, ca_pd_swerling1, os_alpha, os_pfa
from src.proc.theory import ca_pd_swerling0, clustered_ci


def test_alpha_formula():
    assert np.isclose(ca_alpha(32, 1e-4), 32 * (1e-4 ** (-1 / 32) - 1))
    a = os_alpha(32, 24, 1e-4)
    assert np.isclose(os_pfa(a, 32, 24), 1e-4)


def test_pfa_matches_design_iid():
    """Noise-only Pfa within a 4-sigma clustered interval of the design (fast check)."""
    rng = np.random.default_rng(3)
    x = rng.exponential(1.0, (3000, 1024))
    det, thr = ca_cfar_1d(x, 16, 2, 1e-3)
    per = det.sum(1)
    tested = np.isfinite(thr).sum(1)
    lo, hi = clustered_ci(per, tested, z=4.0)
    assert lo <= 1e-3 <= hi


def test_swerling1_matches_closed_form():
    rng = np.random.default_rng(4)
    n, pfa, trials = 32, 1e-3, 40000
    a = ca_alpha(n, pfa)
    for snr_db in (5, 12, 18):
        snr = 10 ** (snr_db / 10)
        train = rng.exponential(1.0, (trials, n)).sum(1)
        s = np.sqrt(snr) * (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)
        nz = (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)
        pd = np.mean(np.abs(s + nz) ** 2 > a / n * train)
        th = ca_pd_swerling1(snr, n, pfa)
        assert abs(pd - th) < 4 * np.sqrt(th * (1 - th) / trials) + 1e-3


def test_swerling0_above_swerling1_at_high_snr():
    assert ca_pd_swerling0(10 ** 1.8, 32, 1e-4) > ca_pd_swerling1(10 ** 1.8, 32, 1e-4)


def test_splits_have_no_shared_instances():
    from src.classify.dataset import split_by_instance, split_heldout_config
    metas = []
    for i in range(40):
        kind = "drone" if i < 20 else "bird"
        m = {"id": i, "label": int(kind == "drone"), "kind": kind}
        m["f_rot" if kind == "drone" else "f_flap"] = 60 + i * 3 if kind == "drone" else 2 + (i - 20) * 0.4
        metas.append(m)
    ids = np.repeat(np.arange(40), 3)
    for s in (split_by_instance(ids, metas, 0), split_heldout_config(ids, metas)):
        tr, va, te = (set(ids[s[k]]) for k in ("train", "val", "test"))
        assert not (tr & va) and not (tr & te) and not (va & te)
        assert len(tr) and len(te)

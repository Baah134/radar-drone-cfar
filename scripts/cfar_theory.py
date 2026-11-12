"""CFAR theory checks (M2).

1. Pfa, i.i.d. exponential noise: 1D CA-CFAR (N_TRAIN per side, N_GUARD per side) on
   long noise lines, design Pfa 1e-3, 1e-4, 1e-5; measured rate with 95% Wilson CI.
2. Pfa through the full chain: noise-only FMCW frames -> Hann windows -> range/Doppler
   FFT -> |.|^2 -> 1D CA-CFAR along range on every Doppler row. Windowing correlates
   neighbouring range cells, so this need not match the i.i.d. design exactly.
3. Pd vs SNR at Pfa 1e-4 (cell level): Swerling 0 and Swerling 1 CUT against
   exponential training cells, vs theory (closed form SW1, numerical SW0).
Writes results/cfar_theory.json, figures/pfa_check, figures/pd_vs_snr.
Usage: python -m scripts.cfar_theory [--quick]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.proc.cfar import ca_cfar_1d, ca_pd_swerling1
from src.proc.theory import ca_pd_swerling0, clustered_ci, wilson

N_TRAIN, N_GUARD = 16, 2
N = 2 * N_TRAIN
PFAS = [1e-3, 1e-4, 1e-5]
SEED = 2026


def pfa_iid(pfa, cells, rng, line=4096):
    h = N_TRAIN + N_GUARD
    per_line = line - 2 * h
    lines = int(np.ceil(cells / per_line))
    per = []
    chunk = 2000
    for s in range(0, lines, chunk):
        x = rng.exponential(1.0, (min(chunk, lines - s), line))
        det, _ = ca_cfar_1d(x, N_TRAIN, N_GUARD, pfa)
        per.extend(det.sum(axis=1).tolist())
    k, tested = int(sum(per)), len(per) * per_line
    lo, hi = clustered_ci(per, [per_line] * len(per))
    wl, wh = wilson(k, tested)
    return {"design": pfa, "cells": tested, "lines": len(per), "false_alarms": k, "measured": k / tested,
            "ci95": [lo, hi], "ci_method": "line-clustered (cells in a line share training cells)",
            "design_inside_ci": bool(lo <= pfa <= hi), "naive_binomial_ci95": [wl, wh],
            "design_inside_naive_ci": bool(wl <= pfa <= wh)}


def pfa_chain(pfa, frames, rng, window="hann"):
    from src.proc.chain import rd_map
    from src.radar.params import FMCW
    from src.radar.sim import beat_signal
    R = FMCW()
    per, sizes = [], []
    for _ in range(frames):
        p = rd_map(beat_signal([], R, rng), window, "none")
        det, thr = ca_cfar_1d(p, N_TRAIN, N_GUARD, pfa)
        per.append(int(det.sum()))
        sizes.append(int(np.isfinite(thr).sum()))
    k, tested = int(sum(per)), int(sum(sizes))
    lo, hi = clustered_ci(per, sizes)
    return {"design": pfa, "window": window, "ci_method": "frame-clustered", "frames": frames, "cells": tested, "false_alarms": k, "measured": k / tested,
            "ci95": [lo, hi], "design_inside_ci": bool(lo <= pfa <= hi)}


def pd_curve(swerling, snrs_db, trials, pfa, rng):
    from src.proc.cfar import ca_alpha
    a = ca_alpha(N, pfa)
    out = []
    for sdb in snrs_db:
        snr = 10 ** (sdb / 10)
        train = rng.exponential(1.0, (trials, N)).sum(1)
        n = (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)
        if swerling == 0:
            s = np.sqrt(snr) * np.exp(1j * rng.uniform(0, 2 * np.pi, trials))
        else:
            s = np.sqrt(snr) * (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)
        cut = np.abs(s + n) ** 2
        k = int(np.sum(cut > a / N * train))
        th = ca_pd_swerling0(snr, N, pfa) if swerling == 0 else ca_pd_swerling1(snr, N, pfa)
        lo, hi = wilson(k, trials)
        out.append({"snr_db": float(sdb), "trials": trials, "pd_sim": k / trials, "ci95": [lo, hi],
                    "pd_theory": float(th), "theory_inside_ci": bool(lo <= th <= hi)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    scale = 0.05 if a.quick else 1.0
    res = {"n_train_per_side": N_TRAIN, "n_guard_per_side": N_GUARD, "n_total": N, "seed": SEED}
    # cells: ~ 400 expected false alarms at each design Pfa
    res["pfa_iid"] = [pfa_iid(p, int(400 / p * scale), rng) for p in PFAS]
    res["pfa_chain"] = [pfa_chain(p, max(5, int((400 / p) / 28000 * scale)), rng) for p in PFAS]
    # control: rectangular windows do not correlate neighbouring cells
    res["pfa_chain_rect"] = [pfa_chain(p, max(5, int((400 / p) / 28000 * scale)), rng, "rect") for p in PFAS]
    snrs = np.arange(0, 26, 1.0)
    trials = 2000 if a.quick else 20000
    res["pd"] = {f"swerling{s}": pd_curve(s, snrs, trials, 1e-4, rng) for s in (0, 1)}
    res["wall_s"] = time.time() - t0
    Path("results").mkdir(exist_ok=True)
    out = Path("results/cfar_theory_quick.json" if a.quick else "results/cfar_theory.json")
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for r in res["pfa_iid"] + res["pfa_chain"] + res["pfa_chain_rect"]:
        print(f"Pfa design {r['design']:.0e}: measured {r['measured']:.3e} CI [{r['ci95'][0]:.2e}, {r['ci95'][1]:.2e}] "
              f"({r['false_alarms']} / {r['cells']}) inside={r['design_inside_ci']}")
    for k, v in res["pd"].items():
        inside = sum(x["theory_inside_ci"] for x in v)
        md = max(abs(x["pd_sim"] - x["pd_theory"]) for x in v)
        print(f"{k}: theory inside 95% CI at {inside}/{len(v)} SNR points; max |sim - theory| = {md:.4f}")
    if not a.quick:
        figures(res)


def figures(res):
    from scripts.style import COL_W, save, setup
    plt = setup()
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for key, lab, mk in (("pfa_iid", "i.i.d. exponential noise", "o"), ("pfa_chain", "full FMCW chain, Hann", "s"),
                         ("pfa_chain_rect", "full FMCW chain, rectangular", "^")):
        d = [r["design"] for r in res[key]]
        m = [r["measured"] for r in res[key]]
        lo = [r["measured"] - r["ci95"][0] for r in res[key]]
        hi = [r["ci95"][1] - r["measured"] for r in res[key]]
        ax.errorbar(d, m, yerr=[lo, hi], fmt=mk, ms=4, capsize=2, label=lab)
    ax.plot([1e-5, 1e-3], [1e-5, 1e-3], "k:", lw=0.8, label="measured = design")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("design Pfa")
    ax.set_ylabel("measured false-alarm rate (95% CI)")
    ax.set_title(f"1D CA-CFAR, N = {res['n_total']}, guard {res['n_guard_per_side']}")
    ax.legend(fontsize=6)
    save(fig, "figures/pfa_check")

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    for key, col in (("swerling0", "C0"), ("swerling1", "C1")):
        v = res["pd"][key]
        s = [x["snr_db"] for x in v]
        ax.plot(s, [x["pd_theory"] for x in v], "-", color=col, lw=1, label=f"{key} theory")
        ax.errorbar(s, [x["pd_sim"] for x in v],
                    yerr=[[x["pd_sim"] - x["ci95"][0] for x in v], [x["ci95"][1] - x["pd_sim"] for x in v]],
                    fmt="o", ms=2.5, color=col, capsize=1, label=f"{key} simulation")
    ax.set_xlabel("cell SNR (dB)")
    ax.set_ylabel("Pd (Pfa = 1e-4)")
    ax.legend(fontsize=6)
    save(fig, "figures/pd_vs_snr")


if __name__ == "__main__":
    main()

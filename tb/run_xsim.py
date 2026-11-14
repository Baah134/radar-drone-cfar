"""Run the RTL testbench in Vivado xsim and compare against the bit-exact Python model.

Usage (from repo root): python -m tb.run_xsim [--suite directed random realdata]
Each test: write vectors -> xvlog/xelab/xsim (batch, no project) -> read decisions ->
compare (index, det) lists with src/hw_model/cfar_fixed.py. Pass = identical lists and no
testbench assertion errors. Writes results/rtl_verification.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from src.hw_model.cfar_fixed import cfar_line, quantize_alpha

ROOT = Path(__file__).resolve().parents[1]
VIVADO_BIN = Path(os.environ.get("VIVADO_BIN", r"C:/AMDDesignTools/2025.2/Vivado/bin"))
WORK = ROOT / "sim_build"
DATA_W = 16
MAXV = (1 << DATA_W) - 1


def tool(name):
    exe = VIVADO_BIN / (name + (".bat" if os.name == "nt" else ""))
    if not exe.exists():
        raise SystemExit(f"{exe} not found; set VIVADO_BIN")
    return str(exe)


def run(cmd, cwd):
    if os.name == "nt":
        # .bat wrappers split unquoted arguments at '=' -> quote every argument
        cmd = " ".join(f'"{c}"' for c in cmd)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=(os.name == "nt"))
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return r.stdout


_BUILT = {}


def build(n_train, n_guard, stall_pct=20, seed=1):
    """Compile + elaborate once per parameter set; returns the snapshot work dir."""
    key = (n_train, n_guard, stall_pct, seed)
    if key in _BUILT:
        return _BUILT[key]
    d = WORK / os.environ.get("CFAR_WORK_TAG", "") / f"nt{n_train}_ng{n_guard}_st{stall_pct}_s{seed}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    run([tool("xvlog"), Path(os.environ.get("CFAR_RTL", ROOT / "rtl" / "cfar_ca_1d.v")).as_posix()], d)
    run([tool("xvlog"), "--sv", (ROOT / "tb" / "tb_cfar_ca_1d.sv").as_posix()], d)
    run([tool("xelab"), "tb_cfar_ca_1d", "-s", "snap", "-timescale", "1ns/1ps",
         "-generic_top", f"N_TRAIN={n_train}", "-generic_top", f"N_GUARD={n_guard}",
         "-generic_top", f"STALL_PCT={stall_pct}", "-generic_top", f"SEED={seed}"]
        + (["-debug", "typical"] if os.environ.get("CFAR_DEBUG") else []), d)
    _BUILT[key] = d
    return d


def simulate(lines, n_train, n_guard, alpha_q, stall_pct=20, seed=1):
    d = build(n_train, n_guard, stall_pct, seed)
    vin, vout = d / "vec_in.txt", d / "vec_out.txt"
    with open(vin, "w") as f:
        f.write(f"ALPHA {alpha_q:x}\n")
        for l in lines:
            for j, v in enumerate(l):
                f.write(f"{1 if j == 0 else 0} {int(v):x}\n")
    if vout.exists():
        vout.unlink()
    out = run([tool("xsim"), "snap", "-R"], d)
    msgs = [l for l in out.splitlines() if "TB_DONE" in l or "ERROR" in l]
    def _int(v):            # X/Z from a broken design counts as a mismatching value (-1)
        try:
            return int(v)
        except ValueError:
            return -1
    rows = [tuple(_int(v) for v in s.split()) for s in vout.read_text().split("\n") if s.strip()]
    return rows, msgs


def expected(lines, n_train, n_guard, alpha_q):
    rows = []
    for l in lines:
        idx, det, _, _ = cfar_line(l, n_train, n_guard, alpha_q)
        rows += list(zip(idx.tolist(), det.astype(int).tolist()))
    return rows


def check(name, lines, n_train, n_guard, alpha_q, results, stall_pct=20, seed=1):
    t = time.time()
    got, msgs = simulate(lines, n_train, n_guard, alpha_q, stall_pct, seed)
    exp = expected(lines, n_train, n_guard, alpha_q)
    errs = [m for m in msgs if "ERROR" in m]
    mism = sum(1 for a, b in zip(got, exp) if a != b) + abs(len(got) - len(exp))
    ok = mism == 0 and not errs
    results.append({"test": name, "n_train": n_train, "n_guard": n_guard, "alpha_q": alpha_q,
                    "stall_pct": stall_pct, "tb_seed": seed, "lines": len(lines),
                    "cells": int(sum(len(l) for l in lines)), "decisions": len(exp),
                    "detections": int(sum(e[1] for e in exp)), "mismatches": mism, "tb_errors": errs,
                    "pass": ok, "wall_s": round(time.time() - t, 1)})
    print(f"{'PASS' if ok else 'FAIL'} [{n_train},{n_guard}] {name}: {len(exp)} decisions, "
          f"{sum(e[1] for e in exp)} detections, {mism} mismatches {errs[:1]}", flush=True)
    return ok


def directed(nt, ng, aq):
    W = 2 * (nt + ng) + 1
    L = 128
    z = np.zeros(L, np.int64)
    spike = z.copy()
    spike[60] = 5000
    two = z.copy() + 50
    two[60] = 4000
    two[60 + ng + 2] = 4000                       # second target inside the first one's training window
    edge = np.full(L, 40, np.int64)
    edge[64:] = 4000                              # clutter edge
    edge[70] = 30000                              # target just past the edge
    maxv = np.full(L, MAXV, np.int64)
    maxspike = np.full(L, MAXV // 2, np.int64)
    maxspike[50] = MAXV
    ramp = np.arange(L, dtype=np.int64) * 500 % (MAXV + 1)
    short = [np.full(W - 1, 100, np.int64), np.full(W, 100, np.int64), np.arange(W + 1, dtype=np.int64) * 7,
             np.full(W + 2, 9, np.int64)]
    return [("zeros", [z], aq), ("single_spike", [spike], aq), ("two_close_spikes", [two], aq),
            ("clutter_edge", [edge], aq), ("all_max", [maxv], aq), ("max_spike", [maxspike], aq),
            ("max_alpha_on_max_data", [maxv, maxspike], (1 << 17) - 1), ("alpha_zero", [ramp, spike], 0),
            ("line_lengths_W-1_W_W+1_W+2", short, aq),
            ("many_lines_back_to_back", [spike, edge, two, ramp, z, maxspike] * 3, aq)]


def random_lines(nt, ng, n_lines, rng):
    W = 2 * (nt + ng) + 1
    lines = []
    for _ in range(n_lines):
        L = int(rng.integers(W - 2, 300))
        kind = rng.integers(0, 4)
        if kind == 0:
            l = rng.integers(0, MAXV + 1, L)
        elif kind == 1:
            l = np.minimum(MAXV, np.floor(rng.exponential(64, L)))
        elif kind == 2:
            l = np.minimum(MAXV, np.floor(rng.exponential(64, L)))
            pos = rng.integers(0, L, max(1, L // 20))
            l[pos] = rng.integers(1000, MAXV + 1, pos.size)
        else:
            l = np.where(rng.random(L) < 0.5, MAXV, rng.integers(0, 100, L))
        lines.append(np.asarray(l, np.int64))
    return lines


def real_lines(rng, n_frames=4):
    """Range lines (Doppler rows) of simulated scenes: point targets, with and without clutter."""
    from src.hw_model.cfar_fixed import quantize_power
    from src.proc.chain import rd_map
    from src.radar.params import FMCW
    from src.radar.sim import amp_for_snr, beat_signal, ground_clutter
    R = FMCW()
    lines = []
    for f in range(n_frames):
        tg = [{"r0": float(rng.uniform(10, 140)), "v": float(rng.uniform(-25, 25)),
               "amp": amp_for_snr(float(rng.uniform(5, 35)), R)} for _ in range(4)]
        clut = ground_clutter(R, rng, cnr_db=20.0) if f % 2 else []
        p = rd_map(beat_signal(tg + clut, R, rng), "hann", "mean" if f % 2 else "none")
        q = quantize_power(p)
        lines += [q[d] for d in range(q.shape[0])]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", nargs="*", default=["directed", "random", "realdata"])
    ap.add_argument("--random-lines", type=int, default=2000)
    ap.add_argument("--params", default="8,2;16,2;32,4")
    a = ap.parse_args()
    from src.proc.cfar import ca_alpha
    rng = np.random.default_rng(7)
    results = []
    for pr in a.params.split(";"):
        nt, ng = map(int, pr.split(","))
        aq = quantize_alpha(ca_alpha(2 * nt, 1e-4))
        if "directed" in a.suite:
            for name, lines, alpha in directed(nt, ng, aq):
                check(f"directed/{name}", lines, nt, ng, alpha, results)
        if "random" in a.suite:
            for stall, seed in ((0, 1), (30, 2)):
                lines = random_lines(nt, ng, a.random_lines // 2, rng)
                alpha = int(rng.integers(0, 1 << 17))
                check(f"random/{len(lines)}_lines_stall{stall}_random_alpha", lines, nt, ng, alpha, results, stall, seed)
                check(f"random/{len(lines)}_lines_stall{stall}_design_alpha", lines, nt, ng, aq, results, stall, seed)
        if "realdata" in a.suite:
            check("realdata/simulated_rd_rows", real_lines(rng), nt, ng, aq, results)
    ok = all(r["pass"] for r in results)
    Path("results").mkdir(exist_ok=True)
    Path("results/rtl_verification.json").write_text(json.dumps(
        {"simulator": "Vivado 2025.2 xsim (batch)", "reference": "src/hw_model/cfar_fixed.py",
         "all_pass": ok, "n_tests": len(results), "total_decisions": sum(r["decisions"] for r in results),
         "total_cells": sum(r["cells"] for r in results), "tests": results}, indent=2), encoding="utf-8")
    print(f"ALL PASS: {ok}; {len(results)} tests; {sum(r['decisions'] for r in results)} decisions compared")


if __name__ == "__main__":
    main()

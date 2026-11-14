"""Dump a VCD from xsim for a short directed test and render a waveform figure.
Usage: python -m tb.waveform   -> figures/rtl_waveform.{png,pdf}, results/rtl_waveform.vcd (gitignored)
Test: N_TRAIN=4, N_GUARD=1 (W=11) on two short lines with a spike, no stalls except two
inserted idle cycles, so the 3-cycle latency and the edge rule are visible.
"""
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_vcd(path, names):
    txt = Path(path).read_text()
    ids, width = {}, {}
    for m in re.finditer(r"\$var\s+\w+\s+(\d+)\s+(\S+)\s+(\S+)(?:\s+\[[^\]]*\])?\s+\$end", txt):
        w, code, name = m.groups()
        if name in names and name not in ids.values():
            ids[code] = name
            width[name] = int(w)
    body = txt[txt.find("$enddefinitions"):]
    t = 0
    val = {n: 0 for n in names}
    rows = []
    for line in body.splitlines()[1:]:
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            rows.append((t, dict(val)))
            t = int(line[1:])
        elif line[0] in "01xzXZ" and line[1:] in ids:
            val[ids[line[1:]]] = 1 if line[0] == "1" else 0
        elif line[0] in "bB":
            v, code = line[1:].split()
            if code in ids:
                val[ids[code]] = int(v.replace("x", "0").replace("z", "0"), 2)
    rows.append((t, dict(val)))
    return rows


def main():
    from tb import run_xsim as X
    nt, ng = 4, 1
    os.environ["CFAR_WORK_TAG"] = "wave"
    os.environ["CFAR_DEBUG"] = "1"
    d = X.build(nt, ng, stall_pct=0, seed=1)
    l1 = np.full(16, 10, np.int64)
    l1[8] = 900
    l2 = np.full(12, 10, np.int64)
    l2[5] = 900
    vin = d / "vec_in.txt"
    with open(vin, "w") as f:
        f.write(f"ALPHA {X.quantize_alpha(3.0):x}\n")
        for l in (l1, l2):
            for j, v in enumerate(l):
                f.write(f"{1 if j == 0 else 0} {int(v):x}\n")
    tcl = d / "wave.tcl"
    tcl.write_text("open_vcd wave.vcd\nlog_vcd [get_objects -r *]\nrun all\nclose_vcd\nquit\n")
    X.run([X.tool("xsim"), "snap", "-tclbatch", "wave.tcl"], d)
    names = ["clk", "i_valid", "i_sol", "i_data", "o_valid", "o_idx", "o_det"]
    rows = parse_vcd(d / "wave.vcd", names)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rtl_waveform.vcd").write_bytes((d / "wave.vcd").read_bytes())
    from scripts.style import DBL_W, save, setup
    plt = setup()
    # sample on rising clock edges (value just before each edge)
    edges = [r for r in rows if r[1]["clk"] == 1]
    edges = edges[2:50]
    fig, axs = plt.subplots(len(names) - 1, 1, figsize=(DBL_W, 3.2), sharex=True)
    cyc = np.arange(len(edges))
    for ax, n in zip(axs, names[1:]):
        v = np.array([e[1][n] for e in edges])
        ax.step(cyc, v, where="post", lw=1)
        if n in ("i_data", "o_idx"):
            for c, val in zip(cyc, v):
                if n == "o_idx" and not edges[c][1]["o_valid"]:
                    continue
                ax.text(c + 0.5, v.max() * 0.4 if v.max() else 0.4, str(val), fontsize=4, ha="center")
        ax.set_ylabel(n, rotation=0, ha="right", fontsize=6)
        ax.set_yticks([])
    axs[-1].set_xlabel("clock cycle")
    axs[0].set_title("cfar_ca_1d, N_TRAIN=4, N_GUARD=1 (W=11): two lines with one spike each; "
                     "decisions start once a window is full, 3-cycle latency", fontsize=6)
    save(fig, "figures/rtl_waveform")
    print("waveform:", len(edges), "cycles")


if __name__ == "__main__":
    main()

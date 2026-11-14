"""Open-source synthesis ESTIMATES with Yosys synth_xilinx (7-series family, xc7).

Usage: python -m synth.run_yosys
Yosys runs as yowasp-yosys (WebAssembly build from PyPI) inside the venv. For each
(N_TRAIN, N_GUARD) it runs: read_verilog; chparam; synth_xilinx -family xc7 -top
cfar_ca_1d; stat -tech xilinx. Cell counts are pre-place-and-route mapping results:
no timing, no Fmax, not a device implementation. Writes synth/reports/yosys_*.txt and
results/synth_yosys.json.
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARAMS = [(8, 2), (16, 2), (32, 4)]


def yosys_exe():
    for c in ("yowasp-yosys", "yosys"):
        p = shutil.which(c) or shutil.which(str(ROOT / ".venv" / "Scripts" / (c + ".exe")))
        if p:
            return p
    raise SystemExit("yosys not found (pip install yowasp-yosys)")


def parse_stat(txt):
    """Cell counts from the last 'stat' block."""
    i = txt.rfind("=== cfar_ca_1d ===")
    block = txt[i:] if i >= 0 else txt[txt.rfind("Number of cells"):]
    cells = {}
    for m in re.finditer(r"^\s+(\d+)\s+(?:[\d.]+\s+)?\$?([A-Za-z0-9_$]+)\s*$", block, re.M):
        if m.group(2) not in ("wires", "bits", "ports", "cells"):
            cells[m.group(2)] = int(m.group(1))
    if not cells:   # older/newer stat layouts: "   <name>   <count>"
        for m in re.finditer(r"^\s+([A-Za-z0-9_$]+)\s+(\d+)\s*$", block, re.M):
            cells[m.group(1)] = int(m.group(2))
    lut = sum(v for k, v in cells.items() if re.fullmatch(r"LUT[1-6]", k))
    ff = sum(v for k, v in cells.items() if re.fullmatch(r"FD[A-Z]*", k))
    srl = sum(v for k, v in cells.items() if k.startswith("SRL"))
    dsp = sum(v for k, v in cells.items() if k.startswith("DSP48"))
    bram = sum(v for k, v in cells.items() if k.startswith("RAMB"))
    carry = sum(v for k, v in cells.items() if k.startswith("CARRY"))
    return {"LUT": lut, "FF": ff, "SRL": srl, "DSP48": dsp, "BRAM": bram, "CARRY4": carry, "cells": cells}


def main():
    exe = yosys_exe()
    out_dir = ROOT / "synth" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    rtl = (ROOT / "rtl" / "cfar_ca_1d.v").as_posix()
    res = []
    ver = subprocess.run([exe, "-V"], capture_output=True, text=True).stdout.strip()
    for nt, ng in PARAMS:
        script = (f"read_verilog {rtl}; chparam -set N_TRAIN {nt} -set N_GUARD {ng} cfar_ca_1d; "
                  f"synth_xilinx -family xc7 -top cfar_ca_1d; tee -o synth/reports/_stat.txt stat -tech xilinx")
        r = subprocess.run([exe, "-q", "-p", script], capture_output=True, text=True, cwd=ROOT)
        stat_file = ROOT / "synth" / "reports" / "_stat.txt"
        txt = (stat_file.read_text(encoding="utf-8") if stat_file.exists() else "") + r.stdout + r.stderr
        if stat_file.exists():
            stat_file.unlink()
        (out_dir / f"yosys_nt{nt}_ng{ng}.txt").write_text(txt, encoding="utf-8")
        if r.returncode != 0:
            raise SystemExit(f"yosys failed for N_TRAIN={nt}:\n{txt[-2000:]}")
        s = parse_stat(txt)
        s.update({"n_train": nt, "n_guard": ng, "window": 2 * (nt + ng) + 1})
        res.append(s)
        print(f"N_TRAIN={nt:2d} N_GUARD={ng}: LUT {s['LUT']}, FF {s['FF']}, SRL {s['SRL']}, DSP48 {s['DSP48']}, "
              f"BRAM {s['BRAM']}, CARRY4 {s['CARRY4']}")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "synth_yosys.json").write_text(json.dumps(
        {"tool": ver, "flow": "synth_xilinx -family xc7 (open-source ESTIMATE, no place-and-route, no timing)",
         "results": res}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

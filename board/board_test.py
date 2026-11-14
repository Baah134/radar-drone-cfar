"""On-board test of cfar_ca_1d on a Digilent Basys 3.

Usage (repo root):
  python -m board.board_test gen       # ROM contents + expected decisions (board/build/)
  python -m board.board_test sim       # whole board design in xsim, PC side emulated over UART
  python -m board.board_test build     # Vivado synth/place/route/bitstream (+ util/timing reports)
  python -m board.board_test program   # program the board over JTAG
  python -m board.board_test run --port COM10   # run 'R' and 'S' on the board, compare bit-exact
Writes board/reports/* and results/board_measured.json.
Stimulus (N_TRAIN 16, N_GUARD 2, alpha_q for Pfa 1e-4): range lines from simulated radar
scenes, random lines and the directed lines from tb/run_xsim.py, packed into 16,384 cells.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "board"
BUILD = B / "build"
REP = B / "reports"
VIVADO = Path(os.environ.get("VIVADO_BIN", r"C:/AMDDesignTools/2025.2/Vivado/bin")) / "vivado.bat"
PART = "xc7a35tcpg236-1"
N_CELLS = 16384
NT, NG = 16, 2


def make_lines(seed=2027, n_cells=N_CELLS):
    from tb.run_xsim import directed, random_lines, real_lines
    from src.hw_model.cfar_fixed import quantize_alpha
    from src.proc.cfar import ca_alpha
    rng = np.random.default_rng(seed)
    aq = quantize_alpha(ca_alpha(2 * NT, 1e-4))
    pool = [l for _, ls, a in directed(NT, NG, aq) if a == aq for l in ls]
    radar = real_lines(rng, n_frames=2)
    rng.shuffle(radar)
    rand = random_lines(NT, NG, 400, rng)
    lines, total = [], 0
    for src in (pool, radar[:28], rand):
        for l in src:
            l = np.asarray(l, np.int64)
            if total + len(l) > n_cells:
                break
            lines.append(l)
            total += len(l)
    # pad the remainder with one final noise line so the ROM is exactly n_cells long
    if total < n_cells:
        lines.append(np.minimum(65535, np.floor(rng.exponential(64, n_cells - total))).astype(np.int64))
    return lines, aq


def expected(lines, aq):
    from src.hw_model.cfar_fixed import cfar_line
    rows = []
    for l in lines:
        idx, det, _, _ = cfar_line(l, NT, NG, aq)
        rows += list(zip(idx.tolist(), det.astype(int).tolist()))
    return rows


def write_rom(lines, path):
    with open(path, "w") as f:
        for l in lines:
            for j, v in enumerate(l):
                f.write(f"{(1 << 16) * (j == 0) + int(v):05x}\n")


def cmd_gen(a):
    BUILD.mkdir(parents=True, exist_ok=True)
    lines, aq = make_lines()
    write_rom(lines, BUILD / "cfar_rom.hex")
    exp = expected(lines, aq)
    (BUILD / "expected.json").write_text(json.dumps({"alpha_q": aq, "n_cells": int(sum(map(len, lines))),
                                                      "n_lines": len(lines), "decisions": exp}), encoding="utf-8")
    print(f"ROM: {sum(map(len, lines))} cells, {len(lines)} lines, alpha_q {aq}; expected {len(exp)} decisions, "
          f"{sum(d for _, d in exp)} detections")
    assert aq == 43715, "ALPHA_Q parameter in basys3_cfar_top.v must match"


def compare(got, exp):
    mism = sum(1 for g, e in zip(got, exp) if tuple(g) != tuple(e)) + abs(len(got) - len(exp))
    return mism


def cmd_sim(a):
    from tb.run_xsim import tool, run
    d = BUILD / "sim_top"
    d.mkdir(parents=True, exist_ok=True)
    lines, aq = make_lines(seed=5, n_cells=1500)
    write_rom(lines, d / "rom_small.hex")
    exp = expected(lines, aq)
    run([tool("xvlog"), (ROOT / "rtl" / "cfar_ca_1d.v").as_posix(), (B / "uart.v").as_posix(),
         (B / "basys3_cfar_top.v").as_posix()], d)
    run([tool("xvlog"), "--sv", (B / "tb_top.sv").as_posix()], d)
    res = {}
    for cmd in ("R", "S"):
        run([tool("xelab"), "tb_top", "-s", f"snap_{cmd}", "-generic_top", "N_CELLS=1500",
             "-generic_top", f"CMD=\"{cmd}\""], d)
        out = run([tool("xsim"), f"snap_{cmd}", "-R"], d)
        got = [tuple(map(int, s.split())) for s in (d / "dump.txt").read_text().splitlines() if s.strip()]
        reply = (d / "reply.txt").read_text().split()
        m = compare(got, exp)
        res[cmd] = {"decisions": len(got), "expected": len(exp), "mismatches": m, "reply": reply}
        print(f"sim '{cmd}': {len(got)} decisions, {m} mismatches, reply {reply}", flush=True)
    return res


TCL_BUILD = """
read_verilog {{{rtl}}} {{{uart}}} {{{top}}}
read_xdc {{{xdc}}}
add_files -norecurse cfar_rom.hex
synth_design -top basys3_cfar_top -part {part}
opt_design
place_design
route_design
report_utilization -file {{{rep}/util.rpt}}
report_utilization -hierarchical -file {{{rep}/util_hier.rpt}}
report_timing_summary -file {{{rep}/timing.rpt}}
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set fp [open {{{rep}/wns.txt}} w]; puts $fp $wns; close $fp
write_bitstream -force {{{bit}}}
"""

TCL_PROG = """
open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target
set dev [lindex [get_hw_devices xc7a35t*] 0]
current_hw_device $dev
set_property PROGRAM.FILE {{{bit}}} $dev
program_hw_devices $dev
puts "PROGRAMMED [get_property NAME $dev]"
close_hw_target
quit
"""


def vivado(tcl_text, name):
    BUILD.mkdir(parents=True, exist_ok=True)
    t = BUILD / name
    t.write_text(tcl_text, encoding="utf-8")
    log = BUILD / (name + ".log")
    r = subprocess.run(f'"{VIVADO}" -mode batch -nojournal -log "{log.as_posix()}" '
                       f'-source "{t.as_posix()}"', cwd=BUILD, shell=True, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stdout[-4000:] + r.stderr[-2000:])
    # severities are never ignored: any CRITICAL WARNING fails the build (a missing ROM file
    # once produced a "working" bitstream with an all-zero ROM)
    crit = [l for l in log.read_text(errors="replace").splitlines() if l.startswith("CRITICAL WARNING")]
    if crit:
        raise SystemExit("Vivado CRITICAL WARNING(s): " + " | ".join(crit[:10]))
    return r.stdout


def cmd_build(a):
    REP.mkdir(parents=True, exist_ok=True)
    vivado(TCL_BUILD.format(rtl=(ROOT / "rtl/cfar_ca_1d.v").as_posix(), uart=(B / "uart.v").as_posix(),
                            top=(B / "basys3_cfar_top.v").as_posix(), xdc=(B / "basys3.xdc").as_posix(),
                            part=PART, rom=(BUILD / "cfar_rom.hex").as_posix(), rep=REP.as_posix(),
                            bit=(BUILD / "basys3_cfar.bit").as_posix()), "build.tcl")
    wns = float((REP / "wns.txt").read_text())
    print(f"built; WNS at 100 MHz = {wns:.3f} ns")


def cmd_program(a):
    out = vivado(TCL_PROG.format(bit=(BUILD / "basys3_cfar.bit").as_posix()), "program.tcl")
    print([l for l in out.splitlines() if "PROGRAMMED" in l or "ERROR" in l])


def cmd_run(a):
    import serial
    E = json.loads((BUILD / "expected.json").read_text(encoding="utf-8"))
    exp = [tuple(x) for x in E["decisions"]]
    s = serial.Serial(a.port, 115200, timeout=5)
    time.sleep(0.2)
    s.reset_input_buffer()
    runs = {}
    for cmd in [c for _ in range(a.reps) for c in ("R", "S")]:
        s.write(cmd.encode())
        rep = s.read(17)
        if len(rep) != 17 or rep[0:1] != b"K":
            raise SystemExit(f"bad reply to {cmd}: {rep!r}")
        n = int.from_bytes(rep[1:3], "big")
        cycles = int.from_bytes(rep[3:7], "big")
        n_in = int.from_bytes(rep[7:9], "big") or 65536
        stalls = int.from_bytes(rep[9:13], "big")
        in_cycles = int.from_bytes(rep[13:17], "big")
        s.write(b"D")
        raw = s.read(2 * n)
        got = [((raw[2 * i] << 8 | raw[2 * i + 1]) >> 1, raw[2 * i + 1] & 1) for i in range(len(raw) // 2)]
        m = compare(got, exp)
        runs.setdefault(cmd, []).append({"stall_mode": cmd == "S", "decisions": len(got), "expected_decisions": len(exp),
                     "detections": int(sum(d for _, d in got)), "mismatches": m, "bit_exact": m == 0,
                     "cells_in": n_in, "cycles_first_input_to_last_decision": cycles, "stall_cycles": stalls,
                     "input_cycles_first_to_last": in_cycles,
                     "throughput_cells_per_input_cycle": round(n_in / in_cycles, 4)})
        print(f"board '{cmd}': {len(got)}/{len(exp)} decisions, {m} mismatches; {n_in} cells in {in_cycles} input "
              f"cycles ({stalls} stall cycles)", flush=True)
    s.close()
    wns = float((REP / "wns.txt").read_text())
    util = (REP / "util.rpt").read_text()
    import re
    def grab(name):
        mm = re.search(r"\|\s*" + re.escape(name) + r"\s*\|\s*(\d+)", util)
        return int(mm.group(1)) if mm else None
    rec = {"date": time.strftime("%Y-%m-%d"), "board": "Digilent Basys 3 (xc7a35tcpg236-1)",
           "who": "Prince's board, connected by Prince; built, programmed and run over JTAG/UART by Claude Code",
           "tool": "Vivado 2025.2 (full implementation of the test design, 100 MHz system clock)",
           "clock_MHz": 100, "wns_ns_at_100MHz": wns,
           "test_design_utilization": {"LUT": grab("Slice LUTs"), "FF": grab("Slice Registers"),
                                       "BRAM_tile": grab("Block RAM Tile"), "DSP": grab("DSPs")},
           "stimulus": {"cells": E["n_cells"], "lines": E["n_lines"], "alpha_q": E["alpha_q"],
                        "n_train": NT, "n_guard": NG,
                        "content": "directed lines, range lines from simulated radar scenes, random lines"},
           "repetitions_per_mode": a.reps,
           "all_runs_bit_exact": all(r["bit_exact"] for rs in runs.values() for r in rs),
           "runs": runs, "method": "ROM-fed stream at one cell/clock (R) and with ~25% LFSR stalls (S); "
                                   "decisions captured on chip, read over UART, compared with src/hw_model/cfar_fixed.py"}
    (ROOT / "results" / "board_measured.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("ALL BIT-EXACT:", rec["all_runs_bit_exact"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "sim", "build", "program", "run"])
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    {"gen": cmd_gen, "sim": cmd_sim, "build": cmd_build, "program": cmd_program, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()

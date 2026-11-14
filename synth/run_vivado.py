"""Vivado out-of-context synthesis + place & route of cfar_ca_1d (TOOL REPORTS, not hardware).

Usage: python -m synth.run_vivado [--part xc7a35tcpg236-1] [--period 4.0]
For each (N_TRAIN, N_GUARD): synth_design -mode out_of_context, a clock constraint on
clk, opt/place/route, then utilization and timing reports. The Fmax figure is computed
from the post-route worst negative slack: 1 / (period - WNS). Out-of-context means no
I/O buffers or board-level constraints; it estimates the block itself on that device.
Writes synth/reports/vivado_*.rpt and results/synth_vivado.json.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIVADO_BIN = Path(os.environ.get("VIVADO_BIN", r"C:/AMDDesignTools/2025.2/Vivado/bin"))
PARAMS = [(8, 2), (16, 2), (32, 4)]

TCL = """
read_verilog {rtl}
synth_design -mode out_of_context -top cfar_ca_1d -part {part} -generic N_TRAIN={nt} -generic N_GUARD={ng}
create_clock -name clk -period {period} [get_ports clk]
opt_design
place_design
route_design
report_utilization -file {rep}_util.rpt
report_timing_summary -file {rep}_timing.rpt
set wns [get_property SLACK [get_timing_paths -max_paths 1 -nworst 1 -setup]]
set fp [open {rep}_wns.txt w]
puts $fp $wns
close $fp
"""


def util_counts(txt):
    def grab(name):
        m = re.search(r"\|\s*" + re.escape(name) + r"\s*\|\s*(\d+)", txt)
        return int(m.group(1)) if m else None
    return {"LUT": grab("Slice LUTs"), "LUT_as_logic": grab("LUT as Logic"),
            "LUT_as_shift_register": grab("LUT as Shift Register"), "FF": grab("Slice Registers"),
            "DSP48E1": grab("DSPs"), "BRAM_tile": grab("Block RAM Tile")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="xc7a35tcpg236-1")
    ap.add_argument("--period", type=float, default=4.0)
    a = ap.parse_args()
    out = ROOT / "synth" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    work = ROOT / "synth" / "vivado_work"
    work.mkdir(exist_ok=True)
    res = []
    for nt, ng in PARAMS:
        rep = (out / f"vivado_nt{nt}_ng{ng}").as_posix()
        tcl = work / f"run_nt{nt}.tcl"
        tcl.write_text(TCL.format(rtl=(ROOT / "rtl" / "cfar_ca_1d.v").as_posix(), part=a.part, nt=nt, ng=ng,
                                  period=a.period, rep=rep), encoding="utf-8")
        cmd = f'"{VIVADO_BIN / "vivado.bat"}" -mode batch -nojournal -nolog -source "{tcl.as_posix()}"'
        r = subprocess.run(cmd, cwd=work, capture_output=True, text=True, shell=True)
        if r.returncode != 0:
            raise SystemExit(f"vivado failed for N_TRAIN={nt}:\n{r.stdout[-3000:]}\n{r.stderr[-2000:]}")
        u = util_counts(Path(rep + "_util.rpt").read_text())
        wns = float(Path(rep + "_wns.txt").read_text().strip())
        fmax = 1000.0 / (a.period - wns)
        res.append({"n_train": nt, "n_guard": ng, **u, "clock_period_ns": a.period, "wns_ns": wns,
                    "fmax_estimate_MHz": round(fmax, 1)})
        print(f"N_TRAIN={nt}: {u} WNS {wns:.3f} ns -> Fmax estimate {fmax:.0f} MHz", flush=True)
    ver = subprocess.run(f'"{VIVADO_BIN / "vivado.bat"}" -version', capture_output=True, text=True,
                         shell=True).stdout.splitlines()
    (ROOT / "results" / "synth_vivado.json").write_text(json.dumps(
        {"tool": next((l for l in ver if l.startswith("vivado") or l.startswith("Vivado")), "Vivado"),
         "part": a.part, "flow": "out-of-context synth_design + opt/place/route_design (TOOL REPORT, "
                                 "not measured on hardware; no I/O)", "results": res}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()

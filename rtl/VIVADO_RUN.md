# Running the CFAR block in Vivado yourself (Prince)

Everything in `results/` about resources and timing comes from tools run by Claude Code
(Yosys estimates and out-of-context Vivado reports for xc7a35tcpg236-1). Nothing has run
on a board. These steps reproduce the reports on your machine and take it further.

## 1. Same reports as the repo (out-of-context, no board)

```bash
python -m synth.run_vivado --part xc7a35tcpg236-1 --period 4.0
```
Change `--part` to your board's device (e.g. `xc7a100tcsg324-1` for an Arty A7-100T,
`xc7a35tcpg236-1` for a Basys 3). Reports land in `synth/reports/vivado_*`.

## 2. RTL simulation in the Vivado GUI (waveforms)

1. Create a project (RTL project, no sources), add `rtl/cfar_ca_1d.v` as a design source and
   `tb/tb_cfar_ca_1d.sv` as a simulation source; set `tb_cfar_ca_1d` as the simulation top.
2. Generate vectors: `python -c "from tb.run_xsim import *; ..."` or run `python -m tb.run_xsim`
   once and copy `sim_build/nt16_ng2_st20_s1/vec_in.txt` into the simulation directory.
3. Run behavioral simulation; compare `vec_out.txt` with the Python model
   (`src/hw_model/cfar_fixed.py`).

## 3. On a board (what would make it a hardware result)

The block needs a data source and sink. The simplest test: a top level with a small ROM
of test cells (from `vec_in.txt`), the CFAR, and a counter of detections shown on LEDs or
sent over UART; compare with the Python count. Add an ILA (Integrated Logic Analyzer) on
`o_valid/o_idx/o_det` to capture decisions. Then:
- record utilization and timing from the implemented design (not out-of-context),
- record the clock you ran at and whether outputs matched the model bit for bit.

If you do this, tell me the date, Vivado version, device, clock and method, and I will add
`results/hardware_measured.json` marked as measured by you.

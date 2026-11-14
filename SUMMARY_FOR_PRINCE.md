# Summary for Prince — Project 2 (facts only; write your own words from these)

**Headline (the strongest part):** the verified CFAR hardware block and the theory-matched
detection curves. The classifier is a smaller, idealized add-on.

**What it is.** A simulation study of a 24 GHz FMCW radar for small-drone detection: a full
processing chain, CFAR detection checked against theory, a streaming CA-CFAR hardware block in
Verilog verified bit-exactly against a Python fixed-point model with FPGA resource and timing
estimates, and a micro-Doppler drone-vs-bird classifier with leakage-safe evaluation.

**What you built (with Claude Code).**
- FMCW simulator (dechirped IF signal, ground clutter, drone rotor and bird wing scatterers),
  range/Doppler processing, MTI and mean clutter suppression, CA/OS/2D CFAR.
- Theory checks: false-alarm rate vs design (i.i.d. and through the full chain) and Pd vs SNR
  for Swerling 0 and 1 against closed-form/numerical theory.
- `cfar_ca_1d.v`: one cell per clock, running sums (O(1) per cell), integer compare with one
  DSP48, no-reset SRL-friendly window, 3-cycle latency; bit-exact Python model.
- Verification in Vivado xsim: directed, random (with input stalls) and radar-derived tests;
  overflow assertions; mutation testing of the harness.
- Yosys and Vivado resource/timing reports for N_TRAIN = 8, 16, 32.
- Micro-Doppler dataset with instance IDs, instance and held-out-configuration splits, a
  deliberately leaky split, TinyCNN (simulated INT8) and a feature + logistic-regression baseline.

**Key numbers (all from RESULTS.md).**
- CFAR theory: Swerling 0 theory inside the 95% CI at 26/26 SNR points, Swerling 1 at 25/26;
  Pfa matches design for i.i.d. noise and for the chain with rectangular windows.
- Hann windowing raises the real chain's false-alarm rate 1.7–3.6x above design.
- Two close targets: CA-CFAR Pd 0.90 → 0.67 when the second target is in the training
  window; OS-CFAR stays at 0.89. At a +15 dB clutter edge both mask a target on the low side.
- RTL: bit-exact on 45 tests, 1,868,856 decisions, 0 mismatches; 7/7 non-equivalent mutants caught.
- Fixed vs float CFAR on realistic frames: 99.96% decision agreement (differences mostly from
  16-bit input saturation in strong clutter).
- Vivado (out-of-context, xc7a35t-1): 122–147 LUTs, 219–243 FFs, 1 DSP48, Fmax estimate
  191–203 MHz for N_TRAIN 8–32. Yosys: 136–142 LUTs, 1 DSP48 (17-bit alpha).
- Classifier (simulated point scatterers; a secondary, idealized study): CNN with 4,778
  parameters reaches 98.7% ± 0.5% on unseen instances (simulated INT8 98.4%); the
  deliberately leaky random-slice split gains only +0.4 points because drones and birds are
  physically very different in this simulation; on unseen rotor/flap rates the CNN drops to
  93.9% ± 3.4% while a 6-feature logistic regression holds 96.3%. Accuracy falls to 93.5% at
  the lowest SNR band (-5..0 dB per pulse).

**Words to avoid.** "deployed on FPGA", "runs at 200 MHz on hardware", "timing closed",
"tested on real radar data", "detects drones" (say "in simulation"). Say "verified in RTL
simulation", "Vivado out-of-context estimate", "simulated".

**Link to your other work.** The bit-exact fixed-point-vs-reference method is the same one you
use for the CREATE Lab accelerators; the leaky-vs-instance split is the same lesson as your
SER audit.

**What hardware validation would need.** Implement on your Artix-7 board with a test ROM and
an ILA or UART readout, record utilization/timing from the full implementation and compare
outputs with the model (rtl/VIVADO_RUN.md). Real radar data (e.g. a 24 GHz radar module)
would be needed to test the detection and classification claims.

**Things to be able to explain.** Beat frequency → range, chirp phase → velocity; the α formula
and Swerling 1 Pd; why Hann windows break the CFAR independence assumption; CA vs OS; running
sums; why 17 bits fit one DSP; what an equivalent mutant is; why random-slice splits leak.

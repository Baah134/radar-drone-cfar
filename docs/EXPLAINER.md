# EXPLAINER — study guide for Prince (Project 2)

Numbers here are quoted from RESULTS.md (generated from results files); if they disagree,
RESULTS.md wins.

## 1. FMCW basics with this project's numbers

- The radar transmits a chirp whose frequency rises linearly by B = 250 MHz over Tc = 100 µs
  (slope S = 2.5 THz/s), starting at 24 GHz (wavelength λ = 12.5 mm).
- **Dechirping:** the echo from range R arrives τ = 2R/c later; mixing it with the
  transmitted chirp leaves a **beat tone** f_b = S·τ = 2SR/c. Range = beat frequency.
  An FFT over the 256 samples of one chirp (the *range FFT*) turns each range into a bin.
- **Range resolution** c/(2B) = 0.60 m. With 256 complex samples across the chirp
  (2.56 MHz complex rate) the maximum range is 256 × 0.60 = 153.5 m.
- **Velocity:** between consecutive chirps a target at speed v moves vTc, which changes the
  echo phase by 4π·vTc/λ. An FFT across the 128 chirps (the *Doppler FFT*) turns that phase
  progression into a velocity bin. Resolution λ/(2·128·Tc) = 0.49 m/s; unambiguous range
  ±λ/(4Tc) = ±31.2 m/s. Faster motions *alias* (wrap around).
- The result is a **range-Doppler map** (128 × 256 cells); squaring the magnitude gives power
  (the square-law detector), in units where noise has mean 1.

## 2. Why small drones are hard

Small radar cross-section (~0.01 m², similar to a bird), slow and low, often hovering, so
they sit close to zero Doppler where ground clutter lives; and birds have similar size and
speed. Detection must work near clutter, and classification must separate drones from birds.

## 3. Clutter suppression

Ground clutter is stationary: it piles up at zero Doppler (with a small spread from wind).
- **Mean subtraction** (remove each range bin's average over the 128 chirps): removes exactly
  zero Doppler, nothing else. Cheap; leaves clutter with any Doppler spread.
- **Two-pulse MTI** y[m] = x[m] − x[m−1]: a high-pass filter in slow time; attenuates near-zero
  Doppler more broadly. Costs a little SNR for slow targets and doubles the noise power
  (normalized out here). **Three-pulse** (x − 2x₋₁ + x₋₂) is steeper still.
- **Window choice** (here Hann): lower sidelobes stop strong clutter leaking into other
  Doppler bins, at the cost of a wider mainlobe and correlated neighbouring cells, which
  matters for CFAR (section 4).

## 4. CFAR: why adaptive thresholds, CA vs OS, and the formulas

A fixed threshold fails because the noise-plus-clutter level changes across the map. CFAR
estimates the local level from **training cells** around the cell under test (skipping
**guard cells** so the target itself doesn't leak in) and multiplies by a factor α.
- **CA-CFAR** averages the N training cells. For exponential (square-law) noise, the sum of N
  cells is Gamma-distributed, and P(cell > (α/N)·sum) = (1 + α/N)^(−N). Setting this equal to
  the design Pfa gives **α = N(Pfa^(−1/N) − 1)**.
- **Swerling 1** target: its power is itself exponential with mean (1 + SNR), so the same
  calculation with the target cell's mean scaled gives **Pd = (1 + α/(N(1+SNR)))^(−N)**.
- **OS-CFAR** sorts the training cells and uses the k-th smallest (here k = 24 of 32). A few
  strong cells (another target) don't move the k-th value much, so OS is robust to
  interfering targets; it is not automatically better at a clutter edge.
- **Findings here:** simulation matches both theories; Hann windowing makes the real chain's
  false-alarm rate 1.7–3.6× the design (neighbouring cells correlated); at a +15 dB clutter
  edge both CA and OS mask a target just on the low side; with a second target in the
  training window CA loses Pd (0.90 → 0.67) while OS keeps it (0.89).

## 5. Micro-Doppler and the aliasing issue

Rotor blades move much faster than the drone (tip speed 2π·f·L, up to 113 m/s here), so their
echoes spread energy over a wide Doppler band that flashes as blades pass; bird wings flap
slowly (2–10 Hz) and create a sinusoidal oscillation around the body line. These signatures
separate drones from birds. The detection waveform only covers ±31 m/s, so blade returns
would **alias**. This project uses a separate **classification dwell** (50 kHz PRF, ±156 m/s)
so the signature is physically clean (spec option (a)).

## 6. The hardware block

- **Running sums:** instead of adding N cells for every output, keep two running sums and per
  clock add the cell entering each training window and subtract the one leaving. Cost is
  O(1) per cell regardless of N (Vivado: 122 → 147 LUTs when N_TRAIN goes 8 → 32).
- **No division:** compare cut·N·2^12 > alpha_q·sum with integers: one DSP48 multiply.
- **Fixed-point formats:** 16-bit unsigned cells; α as Q5.12 (17 bits) because 17 bits fit the
  DSP's 18-bit signed port (18 bits needed 2 DSPs). Widths are chosen so no sum can overflow
  (worst case: every cell at full scale; proved in rtl/README.md, checked by pytest, asserted
  in the testbench every clock).
- **No-reset shift register:** clearing a 16-bit × W register on reset prevents the FPGA's
  shift-register LUTs (SRL) from being used; a 1-bit "filled" flag per position does the same
  job (718 → ~200 flip-flops).
- **Bit-exact verification:** a Python model computes exactly what the RTL should output; the
  testbench streams the same cells into the RTL (with random stalls) and every (index,
  decision) pair must match: 1.87 million decisions, 0 mismatches. **Mutation testing**
  deliberately breaks the RTL in 8 ways to prove the comparison would notice; the one
  "uncaught" mutant changes a bit that can never be 1 (an equivalent mutant).
- **What Yosys/Vivado numbers tell you:** how many LUTs/FFs/DSPs the block needs on a 7-series
  device and (Vivado) an estimated clock limit for the block alone. They do not tell you it
  works on a board with real I/O, clocks and data.

## 7. Leakage in simulated datasets

Each target instance (one set of rotor rate, blade length, flap rate...) produces three
spectrogram slices. They are near copies of each other. If slices are split at random, the
test set contains instances the model has already seen, so it can memorize instance details
instead of learning drone-vs-bird. The primary split puts whole instances in one set; a
held-out-configuration test goes further and tests on rotor and flap rates outside the
training range. The deliberately leaky random-slice split is reported only to measure the
gap. Same lesson as your SER audit (speaker leakage).

## 8. What the results show / do not show

Shows: theory-matched CFAR in simulation; the window and clutter-edge effects; the
classic CA vs OS trade-off; a CFAR block that is bit-exact with its model, robust to stalls,
small (≈120–150 LUTs, 1 DSP) and estimated near 190–200 MHz on an Artix-7 (out-of-context).
Does not show: real radar data, a board implementation, or tracking.

## 9. Likely reviewer questions

1. *Why trust the simulator?* Range/Doppler placement and SNR calibration are unit-tested;
   CFAR Pfa and Pd match closed-form theory before any scene result is used.
2. *Why does the real chain exceed the design Pfa?* Window-correlated cells; a rectangular
   window brings it back to design. A real design would recalibrate α for the window.
3. *Is "bit-exact" meaningful if model and RTL were written by the same person?* The model is
   a direct sum-and-compare (cumulative sums), the RTL uses running sums and a pipeline:
   different algorithms that must agree. Mutation testing shows the harness detects bugs.
4. *Why xsim, not Verilator/cocotb?* Tool availability on this machine (DECISIONS #1); the
   pass criterion is the same.
5. *Is 200 MHz real?* It is Vivado's post-route estimate for the block alone on xc7a35t-1; a
   board design with I/O could differ. Prince can reproduce it (rtl/VIVADO_RUN.md).
6. *Why so many false alarms in clutter?* Strong (30 dB/sample), K-distributed clutter with
   Doppler spread leaks through MTI; plus the Hann baseline (~5 clusters/frame at Pfa 1e-4).
7. *Would the classifier work on real drones?* Unknown: the signatures are idealized point
   scatterers; real data would be the next step.

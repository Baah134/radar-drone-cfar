# DECISIONS — Project 2 (FMCW radar drone detection + verified CFAR hardware)

## Session 1 — 2026-11-11

1. **Tools.** MSYS2's package index on this machine is stale (pacman returned 404 for
   Verilator/Yosys/Icarus); fixing it needs a full `pacman -Syu`, which changes system
   compilers, so it was NOT done without Prince's approval. Used instead:
   Yosys 0.70 from PyPI (`yowasp-yosys`, WebAssembly, inside the venv) for open-source
   synthesis estimates, and Vivado 2025.2 xsim (already installed) for RTL simulation
   with a file-driven testbench compared bit-exactly against the Python golden model.
   cocotb (installed in the venv) needs Icarus or Verilator, so it is not used yet.
   This changes the verification tool, not what is verified (bit-exact RTL vs model).
2. **Python 3.14** venv (3.11 not installed); code avoids 3.12+ syntax.
3. **Waveform:** spec defaults (24 GHz, 250 MHz, 100 us, 256 complex samples, 128 chirps),
   complex (I/Q) sampling: 0.60 m range resolution, 153.5 m max range, +-31.2 m/s,
   0.49 m/s velocity resolution, 12.8 ms frames.
4. **Micro-Doppler: option (a), a separate classification dwell** at 50 kHz PRF
   (+-156 m/s unambiguous; blade tips in the dataset reach 113 m/s), 0.4 s long, simulated
   in slow time at the target's range bin.
5. **Range migration and intra-chirp Doppler ignored:** < 0.38 m motion per frame
   (< 1 range bin) and < 0.5 fast-time bin of Doppler shift.
6. **Noise normalization:** unit-variance complex noise per IF sample; FFTs normalized so
   noise has unit mean power per range-Doppler cell. SNRs are per-cell, post-FFT
   (amp_for_snr includes the window SNR loss; tested to 0.05 dB).
7. **1D CA-CFAR edge rule (software = hardware):** only cells with complete windows on
   both sides are tested; the first/last n_train + n_guard cells of a line never detect.
8. **CFAR theory check settings:** N_TRAIN = 16 per side (N = 32), N_GUARD = 2,
   design Pfa 1e-3/1e-4/1e-5 with ~400 expected false alarms each; Pd at Pfa 1e-4 with
   20,000 trials per SNR point; seed 2026.
9. **Finding: Hann windowing raises the CA-CFAR false-alarm rate 1.7-3.6x above design**
   through the full chain (rectangular windows match the design), because the window
   correlates neighbouring range cells. Kept as a result, not tuned away.
10. **Confidence intervals for Pfa** are line/frame-clustered (cells of a line share
    training cells). They turned out almost identical to the naive binomial ones.
    In the main run 2 of 9 Pfa checks fell outside their 95% CI, both at 1e-3
    (i.i.d. +14%, rectangular chain -7%). A diagnostic with 10x the cells and three new
    seeds (results/cfar_pfa_diagnostic.json) put i.i.d. 1e-3 within 1.4% of design and
    inside the CI 3/3 times, so the miss is a fluctuation, not a bias. Both reported.
11. **Hardware fixed point:** 16-bit unsigned cells; threshold factor Q5.12 (17 bits,
    alpha < 32). An 18-bit factor (Q6.12) mapped the product to 2 DSP48s in Yosys
    (unsigned 18 bits need the DSP's 18-bit port as signed 19); 17 bits use 1. Q5.12
    still covers N = 16 down to Pfa 1e-7 (alpha 27.8). Exact integer compare
    `cut*N*2^12 > alpha_q*sum`: no division, no rounding, so model and RTL are exact.
12. **Input scaling for real data:** floor(power x 64) saturating at 65535: 30.1 dB of
    headroom above the noise mean. On frames with 30 dB/sample clutter, 275 of 401
    float/fixed disagreements (out of 1.13 M decisions, 99.96% agreement) come from
    saturation; a wider DATA_W or a log/shift front end would remove them. Not changed.
13. **Shift register without reset.** First version cleared the 16-bit window register on
    reset (718 FFs in Yosys, no SRLs). Replaced by a 1-bit `filled` flag per position
    that masks pre-reset contents out of the running sums: 177-200 FFs and SRL-mapped
    window. Re-verified in full after the change.
14. **RTL simulator:** Vivado xsim (see #1), file-driven testbench, random stalls (0/20/30%),
    sum-bound assertions every clock. Pass = identical (index, det) lists vs the model.
15. **Mutation testing** of the harness: 8 single-line RTL mutants. The first run exposed a
    harness bug (X outputs crashed the comparison instead of failing it; fixed). One
    mutant (dropping the sum's MSB) is EQUIVALENT: for N = 32, 32*(2^16-1) < 2^21, so
    that bit is always 0. It reveals one spare bit in SUM_W, not a test gap; a 2-bit
    truncation mutant is caught. All 7 non-equivalent mutants caught.
16. **Vivado cross-check:** out-of-context synth + place & route on xc7a35tcpg236-1 with a
    4.0 ns clock target; Fmax reported as 1/(period - WNS). Tool report, not hardware.
17. **Scenes:** the noise-only control gives ~5 false-alarm clusters/frame for 1D CA at
    Pfa 1e-4 (2.8 expected false cells by design; Hann excess, #9), so clutter-induced
    false alarms are the scene numbers minus this baseline.
18. **Classifier dataset revised once (before any classifier result was used):** the first
    version (body SNR -15..+5 dB per pulse, blade amplitude 0.06) made rotor and wing
    signatures invisible in the spectrograms (only the body line; blades ~30 dB below
    the body at <= 5 dB SNR). Revised to body SNR -5..+25 dB per pulse, blade amplitude
    0.08-0.20 of the body, wing amplitude 0.15-0.35, so micro-Doppler is physically
    present. Accuracy is reported per SNR band.
19. **INT8 classifier = simulated** (fake quantization, per-channel int8 weights,
    per-tensor int8 activations calibrated on training data); no integer kernel here.
20. **Scope (2026-11-13, Prince: reduce scope where it helps, keep rigor):** the headline is
    the verified CFAR block + theory-matched detection. The classifier is kept as a
    secondary, idealized study (no further work on it): drone vs bird is easy in this
    simulation (98.7%), the leaky-split gap is small (+0.4 pts) and is reported as such;
    the informative number is the held-out-configuration drop (CNN 93.9% vs features 96.3%).
    Not done: OS-CFAR hardware, cocotb/Verilator flow, real radar data.

## Session 2 - 2026-11-13 (on-board test, Basys 3)

21. **On-board test design** (`board/`): 16,384-cell ROM in block RAM (directed, radar-derived
    and random lines; N_TRAIN 16, N_GUARD 2, alpha_q 43715) streamed into cfar_ca_1d at
    100 MHz; decisions captured in block RAM, read over the USB-UART (115200 baud) and compared
    with the Python model. Modes: one cell per clock, and ~25% LFSR stalls. The whole design
    was first simulated in xsim with the testbench acting as the PC.
22. **Bugs found on the way (none in the CFAR block):**
    (a) the testbench returned from sending a UART byte too late and missed the board's first
        reply byte (a real PC UART is full duplex) - testbench fixed;
    (b) a 4-bit constant for the reply length (16) truncated to 0 - widened to 5 bits;
    (c) **the first bitstream ran with an all-zero ROM**: Tcl braces became part of the ROM
        file path, Vivado logged a CRITICAL WARNING (could not open the $readmem data file)
        and still produced a bitstream. The first board run therefore failed (16,148
        mismatches). Fix: the ROM file is added to the build, and the build script now fails
        on ANY Vivado CRITICAL WARNING.
23. **Result:** 10/10 board runs bit-exact (129,640 decisions); 16,384 cells in 16,384 cycles;
    timing met at 100 MHz with 3.03 ns slack. Not run at higher clocks.

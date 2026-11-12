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

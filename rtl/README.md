# `cfar_ca_1d` — streaming 1D cell-averaging CFAR

Synthesizable Verilog-2001. Bit-exact with `src/hw_model/cfar_fixed.py`.

## Interface

| Port | Dir | Width | Meaning |
|---|---|---|---|
| `clk` | in | 1 | clock |
| `rst` | in | 1 | synchronous reset, active high |
| `alpha_q` | in | `ALPHA_W` | threshold factor, unsigned Q5.12 (`alpha_q = round(alpha * 4096)`), hold stable while streaming |
| `i_valid` | in | 1 | a cell is presented this cycle (may be low at any time: stalls are allowed) |
| `i_sol` | in | 1 | this cell is the first of a new range line |
| `i_data` | in | `DATA_W` | cell power (square-law magnitude), unsigned |
| `o_valid` | out | 1 | a decision is presented |
| `o_idx` | out | `IDX_W` | index (within its line) of the cell under test |
| `o_det` | out | 1 | detection flag |

## Parameters

`DATA_W` (16), `N_TRAIN` training cells per side (16), `N_GUARD` guard cells per side (2),
`ALPHA_W` (17), `ALPHA_FRAC` (12), `IDX_W` (12, lines up to 4096 cells).
Window W = 2·(N_TRAIN + N_GUARD) + 1 cells; N = 2·N_TRAIN training cells.

## Decision rule

`det = cut · N · 2^12  >  alpha_q · (lead_sum + lag_sum)`, i.e. `cut > (alpha_q/4096)/N · sum`:
no division and no rounding. For square-law exponential noise, `alpha = N·(Pfa^(-1/N) − 1)`.

## Architecture and timing

- Shift register `sr[0..W-1]` (newest first), no reset, so synthesis maps it to SRL16/32.
- Running sums: `lead += in − sr[N_TRAIN−1]`, `lag += sr[N_TRAIN+2·N_GUARD] − sr[W−1]`.
  Cost per cell is O(1) and does not grow with N_TRAIN. A 1-bit `filled` flag per position
  (cleared by `rst`) masks pre-reset contents out of the sums.
- 3-stage pipeline (shift + sums → products → compare). **Latency: the decision for a cell
  appears 3 clock cycles after the cell that completes its window** (N_TRAIN + N_GUARD
  cells after the cell under test). Throughput: one cell per clock.
- The `alpha_q · sum` product is ≤ 17 × 24 bits unsigned and fits one DSP48E1.

## Edge handling

A decision is produced only for cells with a complete window on both sides. For a line of
length L, cells 0..N_TRAIN+N_GUARD−1 and L−N_TRAIN−N_GUARD..L−1 get **no decision** (no
`o_valid`); they are never detected. The software CFAR (`src/proc/cfar.py`) and the
fixed-point model use exactly the same rule.

## Fixed-point formats and overflow analysis

| Signal | Width | Bound (all cells at 2^16 − 1) |
|---|---|---|
| `lead_sum`, `lag_sum` | DATA_W + ⌈log2(N_TRAIN+1)⌉ | N_TRAIN·(2^16−1) |
| `sum_all` | DATA_W + ⌈log2(N+1)⌉ | N·(2^16−1) |
| `s1_lhs = cut·N·2^12` | DATA_W + ⌈log2(N+1)⌉ + 12 | (2^16−1)·N·2^12 |
| `s1_rhs = alpha_q·sum` | ALPHA_W + width(sum_all) | (2^17−1)·N·(2^16−1) |

Each declared width holds its bound (tests/test_hw_model.py::test_overflow_bounds checks
N_TRAIN = 8…64). The running-sum updates are modulo 2^width, and the true sum always fits,
so intermediate wrap-around cancels exactly. The testbench also asserts at every clock that
neither running sum exceeds N_TRAIN·(2^16−1).

## Verification status

Verified bit-exact against the Python model in Vivado 2025.2 xsim on directed, random and
radar-derived lines for (N_TRAIN, N_GUARD) = (8, 2), (16, 2), (32, 4), with random input
stalls; see `results/rtl_verification.json`. **On hardware:** on a Digilent Basys 3 at
100 MHz (N_TRAIN 16, N_GUARD 2), 10/10 runs bit-exact (129,640 decisions), one cell per
clock, with and without random stalls; see `results/board_measured.json` and `board/`. Mutation testing (`tb/mutation_test.py`)
checks that the harness catches deliberately injected bugs. Block resource numbers are synthesis
estimates (Yosys) and out-of-context Vivado reports; the ~190-200 MHz Fmax figure was not run on the board.

"""Mutation test of the verification harness: deliberately broken RTL variants must FAIL.

Usage: python -m tb.mutation_test
Each mutant is a one-line change to rtl/cfar_ca_1d.v (written to sim_build/mutants/).
The harness passes the check only if every mutant produces at least one mismatch (or a
testbench assertion) against the bit-exact model on the standard random + directed set.
Writes results/rtl_mutation_test.json.
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MUTANTS = {
    "compare_ge_instead_of_gt": ("o_det   <= s1_valid && (s1_lhs > s1_rhs);", "o_det   <= s1_valid && (s1_lhs >= s1_rhs);"),
    "wrong_lead_tap": ("? sr[N_TRAIN-1]         :", "? sr[N_TRAIN-2]         :"),
    "lag_window_off_by_one": ("? sr[N_TRAIN+2*N_GUARD] :", "? sr[N_TRAIN+2*N_GUARD-1] :"),
    "no_filled_mask_on_lag_out": ("wire [DATA_W-1:0] out_lag  = filled[W-1]               ? sr[W-1]               : {DATA_W{1'b0}};",
                                  "wire [DATA_W-1:0] out_lag  = sr[W-1];"),
    "cut_one_cell_early": ("wire [DATA_W-1:0] cut = sr[HALF];", "wire [DATA_W-1:0] cut = sr[HALF-1];"),
    # EQUIVALENT mutant (expected to be missed): for N = 32 the true sum is at most
    # 32*(2^16-1) = 2,097,120 < 2^21, so bit SUM_W-1 (bit 21) is always 0; dropping it
    # cannot change behaviour. It shows SUM_W has one spare bit, not a test gap.
    "EQUIV_drop_unused_sum_msb": ("s1_rhs   <= alpha_q * sum_all;", "s1_rhs   <= alpha_q * sum_all[SUM_W-2:0];"),
    "truncated_sum_2_bits": ("s1_rhs   <= alpha_q * sum_all;", "s1_rhs   <= alpha_q * sum_all[SUM_W-3:0];"),
    "index_off_by_one": ("s0_idx   <= k_next - HALF;", "s0_idx   <= k_next - HALF + 1;"),
}


def main():
    from src.hw_model.cfar_fixed import quantize_alpha
    from src.proc.cfar import ca_alpha
    from tb import run_xsim as X
    src = (ROOT / "rtl" / "cfar_ca_1d.v").read_text(encoding="utf-8")
    mdir = ROOT / "sim_build" / "mutants"
    mdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    nt, ng = 16, 2
    aq = quantize_alpha(ca_alpha(2 * nt, 1e-4))
    lines = X.random_lines(nt, ng, 300, rng) + [l for _, ls, _ in X.directed(nt, ng, aq) for l in ls]
    out = []
    for name, (old, new) in MUTANTS.items():
        assert src.count(old) == 1, f"mutation anchor not found: {name}"
        f = mdir / f"{name}.v"
        f.write_text(src.replace(old, new), encoding="utf-8")
        os.environ["CFAR_RTL"] = str(f)
        os.environ["CFAR_WORK_TAG"] = f"mut_{name}"
        X._BUILT.clear()
        res = []
        try:
            X.check(f"mutant/{name}", lines, nt, ng, aq, res, stall_pct=20, seed=3)
            caught = not res[-1]["pass"]
            detail = {"mismatches": res[-1]["mismatches"], "tb_errors": res[-1]["tb_errors"][:2]}
        except RuntimeError as e:          # failing to compile also counts as caught
            caught, detail = True, {"error": str(e)[-300:]}
        equiv = name.startswith("EQUIV_")
        out.append({"mutant": name, "equivalent": equiv, "caught": caught, **detail})
        tag = ("CAUGHT" if caught else "MISSED") + (" (equivalent mutant: miss expected)" if equiv else "")
        print(f"{tag} {name} {detail}", flush=True)
    os.environ.pop("CFAR_RTL", None)
    os.environ.pop("CFAR_WORK_TAG", None)
    Path(ROOT / "results" / "rtl_mutation_test.json").write_text(json.dumps(
        {"all_non_equivalent_caught": all(o["caught"] for o in out if not o["equivalent"]),
         "mutants": out}, indent=2), encoding="utf-8")
    print("ALL NON-EQUIVALENT MUTANTS CAUGHT:", all(o["caught"] for o in out if not o["equivalent"]))


if __name__ == "__main__":
    main()

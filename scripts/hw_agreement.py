"""Fixed-point (hardware) CFAR vs floating-point CFAR on realistic range-Doppler maps.

Frames: drone(s) + optional K-distributed clutter, Hann, MTI2 or none, as in the scenes
study. For every range line (Doppler row) both detectors use N_TRAIN 16, N_GUARD 2,
Pfa 1e-4 and the same edge rule:
  float : ca_cfar_1d on the float power map with the exact alpha
  fixed : quantize_power (floor(p*64), saturate at 65535) + alpha_q = round(alpha*4096)
Disagreements are attributed to: input saturation in the window, alpha quantization
(float detector run with the quantized alpha agrees), or input flooring (the rest).
Writes results/hw_agreement.json. Usage: python -m scripts.hw_agreement [--frames 40]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.hw_model.cfar_fixed import cfar_line, quantize_alpha, quantize_power
from src.proc.cfar import ca_alpha, ca_cfar_1d

NT, NG, PFA = 16, 2, 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    a = ap.parse_args()
    from src.proc.chain import rd_map, window_snr_gain
    from src.radar.params import FMCW
    from src.radar.sim import amp_for_snr, beat_signal, ground_clutter
    from src.radar.targets import drone, to_frame_scatterers
    R = FMCW()
    g = window_snr_gain("hann", R.Ns) * window_snr_gain("hann", R.Nc)
    rng = np.random.default_rng(99)
    alpha = ca_alpha(2 * NT, PFA)
    aq = quantize_alpha(alpha)
    tot = {"decisions": 0, "agree": 0, "float_det": 0, "fixed_det": 0, "diff": 0,
           "diff_saturation": 0, "diff_alpha_quant": 0, "diff_flooring": 0}
    for f in range(a.frames):
        tg = []
        for _ in range(int(rng.integers(1, 4))):
            sc, _ = drone(rng, float(rng.uniform(20, 140)), float(rng.choice([-1, 1]) * rng.uniform(2, 20)))
            tg += to_frame_scatterers(sc, amp_for_snr(float(rng.uniform(5, 40)), R, g), R.Tc)
        clut = ground_clutter(R, rng, cnr_db=30.0, k_shape=2.0) if f % 2 else []
        p = rd_map(beat_signal(tg + clut, R, rng), "hann", "mti2" if f % 2 else "none")
        q = quantize_power(p)
        fdet, thr = ca_cfar_1d(p, NT, NG, PFA)
        fdet_q, _ = ca_cfar_1d(p, NT, NG, alpha=aq / 4096)
        for d in range(p.shape[0]):
            idx, det, _, _ = cfar_line(q[d], NT, NG, aq)
            fd = fdet[d, idx]
            tot["decisions"] += idx.size
            tot["agree"] += int((fd == det).sum())
            tot["float_det"] += int(fd.sum())
            tot["fixed_det"] += int(det.sum())
            for j in np.nonzero(fd != det)[0]:
                c = idx[j]
                tot["diff"] += 1
                win = q[d, c - NT - NG:c + NT + NG + 1]
                if (win >= 65535).any():
                    tot["diff_saturation"] += 1
                elif fdet_q[d, c] == det[j]:
                    tot["diff_alpha_quant"] += 1
                else:
                    tot["diff_flooring"] += 1
    tot["agreement_rate"] = tot["agree"] / tot["decisions"]
    tot.update({"frames": a.frames, "alpha": alpha, "alpha_q": aq, "alpha_quantized": aq / 4096,
                "pfa_with_quantized_alpha": float((1 + (aq / 4096) / (2 * NT)) ** (-2 * NT))})
    Path("results/hw_agreement.json").write_text(json.dumps(tot, indent=2), encoding="utf-8")
    print(json.dumps(tot, indent=1))


if __name__ == "__main__":
    main()

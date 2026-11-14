"""Detection in realistic and adversarial conditions (M2/M4 study).

Part A (synthetic range lines, Swerling-1 targets, exponential background, Pfa 1e-4,
N_TRAIN 16/side, guard 2, OS k = 24 of 32):
  edge   : background steps up by +15 dB at cell 128; a target (SNR 20 dB over its LOCAL
           background) sits at offsets -20..+20 cells from the edge; plus false-alarm rate
           per cell as a function of distance to the edge (no target).
  pair   : two equal targets (SNR 20 dB) separated by 1..24 cells; Pd of the first target.
Part B (full FMCW chain): one drone (body + rotor blades) at a random range 20-140 m and
  radial speed 2-20 m/s (either sign) in K-distributed ground clutter (CNR 30 dB per sample,
  shape 2, 0.1 m/s spread). Body SNR per cell 5/10/15/20 dB. Each frame is processed with
  clutter suppression none / mean / mti2 and detected with 1D CA (range), 1D OS (range) and
  2D CA (range x Doppler), all at Pfa 1e-4. Pd: a detection within +-1 range bin and
  +-2 Doppler bins of the body; false alarms: detection clusters farther than 3 range bins
  from the drone (rotor returns at the drone's range are not counted as false alarms).
Writes results/scenes.json, figures/cfar_edge_pair, figures/scene_clutter, figures/rd_maps.
Usage: python -m scripts.scenes [--frames 100]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.proc.cfar import ca_cfar_1d, ca_cfar_2d, cluster, os_cfar_1d
from src.proc.theory import wilson

PFA = 1e-4
NT, NG = 16, 2
K_OS = 24
SEED = 314


def line_cfar(x, kind, chunk=2000):
    if kind == "CA":
        return ca_cfar_1d(x, NT, NG, PFA)[0]
    if x.ndim == 2 and x.shape[0] > chunk:      # OS sorts every window: process in chunks (memory)
        return np.concatenate([os_cfar_1d(x[i:i + chunk], NT, NG, PFA, k=K_OS)[0]
                               for i in range(0, x.shape[0], chunk)])
    return os_cfar_1d(x, NT, NG, PFA, k=K_OS)[0]


def part_edge(rng, trials):
    L, edge, step = 256, 128, 10 ** 1.5
    bg = np.where(np.arange(L) >= edge, step, 1.0)
    offs = list(range(-20, 21, 2))
    res = {"offsets": offs, "pd": {}, "fa_vs_distance": {}}
    for kind in ("CA", "OS"):
        pd = []
        for o in offs:
            c = edge + o
            x = rng.exponential(1.0, (trials, L)) * bg
            amp = bg[c] * 10 ** 2.0
            x[:, c] = np.abs(np.sqrt(amp) * (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)
                             + (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) * np.sqrt(bg[c] / 2)) ** 2
            k = int(line_cfar(x, kind)[:, c].sum())
            pd.append({"offset": o, "pd": k / trials, "ci95": wilson(k, trials)})
        res["pd"][kind] = pd
        x = rng.exponential(1.0, (trials * 4, L)) * bg
        det = line_cfar(x, kind)
        fa = det.mean(axis=0)
        res["fa_vs_distance"][kind] = {str(d): float(fa[edge + d]) for d in range(-24, 25, 2)}
    return res


def part_pair(rng, trials):
    L, c0 = 256, 120
    seps = [1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24]
    res = {"separations": seps, "pd_first": {}}
    for kind in ("CA", "OS"):
        out = []
        for s in seps:
            x = rng.exponential(1.0, (trials, L))
            for c in (c0, c0 + s):
                x[:, c] = np.abs(np.sqrt(100.0) * (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)
                                 + (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)) ** 2
            k = int(line_cfar(x, kind)[:, c0].sum())
            out.append({"separation": s, "pd": k / trials, "ci95": wilson(k, trials)})
        res["pd_first"][kind] = out
    # single-target reference
    x = rng.exponential(1.0, (trials, L))
    x[:, c0] = np.abs(np.sqrt(100.0) * (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)
                      + (rng.standard_normal(trials) + 1j * rng.standard_normal(trials)) / np.sqrt(2)) ** 2
    res["pd_single"] = {kind: float(line_cfar(x, kind)[:, c0].mean()) for kind in ("CA", "OS")}
    return res


def part_control(rng, frames):
    """Noise only (no clutter, no target): false-alarm clusters per frame for each detector,
    to separate the design/window baseline from clutter-induced false alarms."""
    from src.proc.chain import rd_map
    from src.radar.params import FMCW
    from src.radar.sim import beat_signal
    R = FMCW()
    cnt = {"CA-1D": 0, "OS-1D": 0, "CA-2D": 0}
    cells = {}
    for _ in range(frames):
        p = rd_map(beat_signal([], R, rng), "hann", "none")
        for d in cnt:
            if d == "CA-2D":
                det, thr, _ = ca_cfar_2d(p, (8, 4), (2, 2), PFA)
                cells[d] = int(det.shape[0] * (det.shape[1] - 2 * 10))
            else:
                det, thr = (ca_cfar_1d(p, NT, NG, PFA) if d == "CA-1D" else os_cfar_1d(p, NT, NG, PFA, k=K_OS))
                cells[d] = int(np.isfinite(thr).sum())
            cnt[d] += len(cluster(det, p))
    return {d: {"false_alarm_clusters_per_frame": cnt[d] / frames, "tested_cells_per_frame": cells[d],
                "design_expected_false_cells_per_frame": cells[d] * PFA} for d in cnt}


def part_scene(rng, frames, snrs=(5, 10, 15, 20)):
    from src.proc.chain import rd_map, window_snr_gain
    from src.radar.params import FMCW
    from src.radar.sim import amp_for_snr, beat_signal, ground_clutter
    from src.radar.targets import drone, to_frame_scatterers
    R = FMCW()
    g = window_snr_gain("hann", R.Ns) * window_snr_gain("hann", R.Nc)
    sup = ("none", "mean", "mti2")
    dets = ("CA-1D", "OS-1D", "CA-2D")
    tally = {f"{s}|{c}|{d}": {"hits": 0, "frames": 0, "fa": 0} for s in snrs for c in sup for d in dets}
    nd = R.Nc
    for snr in snrs:
        for f in range(frames):
            r0 = float(rng.uniform(20, 140))
            v = float(rng.choice([-1, 1]) * rng.uniform(2, 20))
            sc, _ = drone(rng, r0, v)
            tg = to_frame_scatterers(sc, amp_for_snr(snr, R, g), R.Tc)
            clut = ground_clutter(R, rng, cnr_db=30.0, k_shape=2.0)
            x = beat_signal(tg + clut, R, rng)
            rb = r0 / R.range_res
            for c in sup:
                p = rd_map(x, "hann", c, nfft_doppler=nd)
                db = nd // 2 + v / R.v_res
                for d in dets:
                    if d == "CA-2D":
                        det = ca_cfar_2d(p, (8, 4), (2, 2), PFA)[0]
                    else:
                        det = (ca_cfar_1d if d == "CA-1D" else
                               (lambda q, a, b, pf: os_cfar_1d(q, a, b, pf, k=K_OS)))(p, NT, NG, PFA)[0]
                    reps = cluster(det, p)
                    hit = any(abs(r["range_bin"] - rb) <= 1 and min(abs(r["doppler_bin"] - db), nd - abs(r["doppler_bin"] - db)) <= 2
                              for r in reps)
                    fa = sum(1 for r in reps if abs(r["range_bin"] - rb) > 3)
                    t = tally[f"{snr}|{c}|{d}"]
                    t["hits"] += int(hit)
                    t["frames"] += 1
                    t["fa"] += fa
    out = {}
    for k, t in tally.items():
        lo, hi = wilson(t["hits"], t["frames"])
        out[k] = {"pd": t["hits"] / t["frames"], "ci95": [lo, hi], "false_alarms_per_frame": t["fa"] / t["frames"],
                  "frames": t["frames"]}
    return out


def rd_examples(rng):
    """Range-Doppler maps for figures: drone, bird, clutter; before/after suppression."""
    from src.proc.chain import rd_map, window_snr_gain
    from src.radar.params import FMCW
    from src.radar.sim import amp_for_snr, beat_signal, ground_clutter
    from src.radar.targets import bird, drone, to_frame_scatterers
    R = FMCW()
    g = window_snr_gain("hann", R.Ns) * window_snr_gain("hann", R.Nc)
    sc_d, _ = drone(rng, 60.0, 8.0, f_rot=80)
    sc_b, _ = bird(rng, 95.0, -6.0, f_flap=6)
    tg = to_frame_scatterers(sc_d, amp_for_snr(25, R, g), R.Tc) + to_frame_scatterers(sc_b, amp_for_snr(30, R, g), R.Tc)
    clut = ground_clutter(R, rng, cnr_db=30.0, k_shape=2.0)
    x = beat_signal(tg + clut, R, rng)
    return {c: 10 * np.log10(rd_map(x, "hann", c) + 1e-3) for c in ("none", "mti2")}, R


def figures(res, maps, R):
    from scripts.style import COL_W, DBL_W, save, setup
    plt = setup()
    fig, axs = plt.subplots(1, 2, figsize=(DBL_W, 2.3))
    e = res["edge"]
    for kind, col in (("CA", "C0"), ("OS", "C1")):
        axs[0].plot(e["offsets"], [p["pd"] for p in e["pd"][kind]], "o-", ms=3, color=col, label=f"{kind}-CFAR")
        pr = res["pair"]["pd_first"][kind]
        axs[1].plot([p["separation"] for p in pr], [p["pd"] for p in pr], "o-", ms=3, color=col, label=f"{kind}-CFAR")
        axs[1].axhline(res["pair"]["pd_single"][kind], color=col, ls=":", lw=0.8)
    axs[0].axvline(0, color="k", lw=0.6)
    axs[0].set_xlabel("target position relative to +15 dB clutter edge (cells)")
    axs[0].set_ylabel("Pd (SNR 20 dB over local background)")
    axs[1].set_xlabel("separation of two equal targets (cells)")
    axs[1].set_ylabel("Pd of first target (dotted: single target)")
    for ax in axs:
        ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "figures/cfar_edge_pair")

    sc = res["scene"]
    snrs = sorted({int(k.split("|")[0]) for k in sc})
    fig, axs = plt.subplots(1, 2, figsize=(DBL_W, 2.4))
    for ci, c in enumerate(("none", "mean", "mti2")):
        for di, (d, mk) in enumerate((("CA-1D", "o"), ("OS-1D", "s"), ("CA-2D", "^"))):
            axs[0].plot(snrs, [sc[f"{s}|{c}|{d}"]["pd"] for s in snrs], marker=mk, ms=3, color=f"C{ci}",
                        ls=["-", "--", ":"][di], label=f"{c}, {d}")
            axs[1].plot(snrs, [sc[f"{s}|{c}|{d}"]["false_alarms_per_frame"] for s in snrs], marker=mk, ms=3,
                        color=f"C{ci}", ls=["-", "--", ":"][di])
    axs[0].set_ylabel("Pd (drone body)")
    axs[1].set_ylabel("false-alarm clusters per frame")
    axs[1].set_ylim(bottom=0)
    for ax in axs:
        ax.set_xlabel("drone body SNR per cell (dB)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6, frameon=False,
               bbox_to_anchor=(0.5, -0.3))
    fig.suptitle("Drone in K-distributed ground clutter (CNR 30 dB/sample)", y=1.02)
    save(fig, "figures/scene_clutter")

    fig, axs = plt.subplots(1, 2, figsize=(DBL_W, 2.6), sharey=True)
    ext = [0, R.Ns * R.range_res, -R.v_max, R.v_max]
    for ax, (c, m) in zip(axs, maps.items()):
        im = ax.imshow(m, aspect="auto", origin="lower", extent=ext, vmin=-5, vmax=55, cmap="viridis")
        ax.set_title({"none": "no clutter suppression", "mti2": "two-pulse MTI"}[c])
        ax.set_xlabel("range (m)")
    axs[0].set_ylabel("radial velocity (m/s)")
    fig.colorbar(im, ax=axs, label="power (dB over noise)")
    arrow = {"arrowstyle": "->", "color": "w", "lw": 0.8}
    for ax in axs:
        ax.annotate("drone", (60, 8), (70, 22), fontsize=7, color="w", arrowprops=arrow)
        ax.annotate("bird", (95, -6), (108, -20), fontsize=7, color="w", arrowprops=arrow)
    save(fig, "figures/rd_maps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--trials", type=int, default=5000)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    res = {"pfa": PFA, "n_train": NT, "n_guard": NG, "k_os": K_OS, "seed": SEED, "trials_line": a.trials,
           "frames_per_snr": a.frames}
    res["edge"] = part_edge(rng, a.trials)
    res["pair"] = part_pair(rng, a.trials)
    print(f"line studies done {time.time() - t0:.0f}s", flush=True)
    res["control_noise_only"] = part_control(rng, max(20, a.frames // 2))
    print("control", res["control_noise_only"], flush=True)
    res["scene"] = part_scene(rng, a.frames)
    res["wall_s"] = time.time() - t0
    Path("results").mkdir(exist_ok=True)
    Path("results/scenes.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    maps, R = rd_examples(np.random.default_rng(SEED + 1))
    figures(res, maps, R)
    for k, v in res["scene"].items():
        print(k, f"Pd {v['pd']:.2f} FA/frame {v['false_alarms_per_frame']:.2f}")


if __name__ == "__main__":
    main()

"""Micro-Doppler drone-vs-bird classification with leakage-safe evaluation (M5).

Dataset: N_PER_CLASS instances per class, 3 slices each (src/classify/dataset.py), seed 21.
Splits (all report test metrics):
  instance : leakage-safe, whole instances in one split (60/20/20, stratified)  [PRIMARY]
  heldout  : unseen configurations: train on rotor < 100 rev/s and flap < 8 Hz, test on
             rotor 100-120 rev/s and flap 8-10 Hz
  leaky    : DELIBERATELY LEAKY random-slice split (slices of one instance in train and test)
Models: TinyCNN (float and simulated INT8), feature + logistic regression. 3 training seeds.
Metrics: accuracy, macro F1, confusion matrix, accuracy per SNR band.
Writes results/classify.json, figures/spectrograms, figures/classify.
Usage: python -m scripts.classify [--n 400]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.classify.dataset import make_dataset, split_by_instance, split_heldout_config, split_random_slices
from src.classify.models import features, int8_model, n_params, predict, train_cnn, TinyCNN

SEEDS = [0, 1, 2]
DATA_SEED = 21
SNR_BANDS = [(-5, 0), (0, 5), (5, 10), (10, 15), (15, 20), (20, 25)]


def metrics(y, p, snr):
    out = {"accuracy": float((y == p).mean()), "macro_f1": float(f1_score(y, p, average="macro")),
           "confusion": confusion_matrix(y, p, labels=[0, 1]).tolist(), "n": int(len(y))}
    out["acc_by_snr"] = {f"{a}..{b}": float((y[(snr >= a) & (snr < b)] == p[(snr >= a) & (snr < b)]).mean())
                         for a, b in SNR_BANDS if ((snr >= a) & (snr < b)).any()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    X, y, ids, metas = make_dataset(a.n, DATA_SEED)
    snr_inst = {m["id"]: m["snr_db"] for m in metas}
    snr = np.array([snr_inst[i] for i in ids])
    print(f"dataset {X.shape} in {time.time() - t0:.0f}s", flush=True)
    F = features(X)
    res = {"n_instances": len(metas), "n_spectrograms": int(len(X)), "slices_per_instance": 3,
           "data_seed": DATA_SEED, "seeds": SEEDS, "cnn_params": n_params(TinyCNN()), "splits": {}}
    for split_name in ("instance", "heldout", "leaky"):
        per_seed = {"cnn_f32": [], "cnn_int8": [], "features_lr": []}
        for seed in SEEDS:
            if split_name == "instance":
                s = split_by_instance(ids, metas, seed)
            elif split_name == "heldout":
                s = split_heldout_config(ids, metas)
            else:
                s = split_random_slices(len(X), seed)
            tr, va, te = s["train"], s["val"], s["test"]
            if split_name != "leaky":
                assert not (set(ids[tr]) & set(ids[te])) and not (set(ids[va]) & set(ids[te]))
            m = train_cnn(X[tr], y[tr], X[va], y[va], seed)
            per_seed["cnn_f32"].append(metrics(y[te], predict(m, X[te]), snr[te]))
            per_seed["cnn_int8"].append(metrics(y[te], int8_model(m, X[tr])(X[te]), snr[te]))
            lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
            lr.fit(F[tr], y[tr])
            per_seed["features_lr"].append(metrics(y[te], lr.predict(F[te]), snr[te]))
            shared = len(set(ids[tr]) & set(ids[te]))
            print(f"{split_name} seed {seed}: cnn {per_seed['cnn_f32'][-1]['accuracy']:.3f} "
                  f"int8 {per_seed['cnn_int8'][-1]['accuracy']:.3f} lr {per_seed['features_lr'][-1]['accuracy']:.3f} "
                  f"(test instances also in train: {shared})", flush=True)
        summ = {}
        for k, v in per_seed.items():
            acc = [x["accuracy"] for x in v]
            f1 = [x["macro_f1"] for x in v]
            summ[k] = {"accuracy_mean": float(np.mean(acc)), "accuracy_std": float(np.std(acc, ddof=1)),
                       "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": float(np.std(f1, ddof=1)),
                       "per_seed": v}
        res["splits"][split_name] = summ
    res["wall_s"] = time.time() - t0
    Path("results/classify.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    figures(res, X, y, metas, ids)


def figures(res, X, y, metas, ids):
    from scripts.style import COL_W, DBL_W, save, setup
    plt = setup()
    fig, axs = plt.subplots(2, 4, figsize=(DBL_W, 3.8))
    fig.subplots_adjust(hspace=0.55)
    shown = {0: 0, 1: 0}
    for i in range(len(X)):
        lab = int(y[i])
        if shown[lab] >= 4 or i % 3:
            continue
        m = metas[ids[i]]
        ax = axs[1 - lab, shown[lab]]
        ax.imshow(X[i], aspect="auto", origin="lower", cmap="magma", extent=[0, 0.4, -156, 156])
        ttl = (f"drone {m['f_rot']:.0f} rev/s, {m['snr_db']:.0f} dB" if lab else
               f"bird {m['f_flap']:.1f} Hz, {m['snr_db']:.0f} dB")
        ax.set_title(ttl, fontsize=6)
        shown[lab] += 1
        if all(v >= 4 for v in shown.values()):
            break
    for ax in axs[:, 0]:
        ax.set_ylabel("radial velocity (m/s)")
    for ax in axs[1]:
        ax.set_xlabel("time (s)")
    save(fig, "figures/spectrograms")

    fig, axs = plt.subplots(1, 2, figsize=(DBL_W, 2.4))
    names = [("cnn_f32", "CNN float32"), ("cnn_int8", "CNN INT8 (sim.)"), ("features_lr", "features + LR")]
    x = np.arange(3)
    for j, (k, lab) in enumerate(names):
        mu = [res["splits"][s][k]["accuracy_mean"] for s in ("leaky", "instance", "heldout")]
        sd = [res["splits"][s][k]["accuracy_std"] for s in ("leaky", "instance", "heldout")]
        axs[0].bar(x + (j - 1) * 0.27, mu, 0.27, yerr=sd, capsize=2, label=lab)
    axs[0].set_xticks(x, ["leaky random-slice", "by instance", "held-out config"])
    axs[0].set_ylabel("test accuracy (mean ± sd, 3 seeds)")
    axs[0].set_ylim(0.4, 1.02)
    axs[0].legend(fontsize=6, loc="lower left")
    bands = list(res["splits"]["instance"]["cnn_f32"]["per_seed"][0]["acc_by_snr"].keys())
    for k, lab in names:
        v = [np.mean([p["acc_by_snr"].get(b, np.nan) for p in res["splits"]["instance"][k]["per_seed"]]) for b in bands]
        axs[1].plot(bands, v, "o-", ms=3, label=lab)
    axs[1].set_xlabel("body SNR per pulse (dB)")
    axs[1].set_ylabel("accuracy (instance split)")
    axs[1].legend(fontsize=6)
    fig.tight_layout()
    save(fig, "figures/classify")


if __name__ == "__main__":
    main()

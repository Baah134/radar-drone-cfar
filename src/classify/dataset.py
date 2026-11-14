"""Micro-Doppler dataset: drone vs bird spectrograms from the classification dwell.

Each TARGET INSTANCE (unique id) has random physical parameters and is observed for
N_SLICES consecutive dwells of ClassDwell().duration seconds. Every slice becomes one
spectrogram. Slices of the same instance are strongly correlated (same rotor rate, blade
length, flap rate...), which is exactly what a random-slice split would leak.

Instance parameters (drawn per instance, seeded):
  drone: rotor rate 50-120 rev/s, blades 2 or 3, blade length 0.10-0.15 m, aspect
         elevation 5-40 deg, radial speed +-(1-15) m/s
  bird : flap rate 2-10 Hz, span 0.3-1.0 m, radial speed +-(1-15) m/s
  body SNR per pulse: uniform -5 .. +25 dB; blade point amplitude 0.08-0.20 of the body (-14..-22 dB),
  wing point amplitude 0.15-0.35 (revised after a first version made micro-Doppler invisible,
  see docs/DECISIONS.md)
STFT: Hann window 256 pulses (5.1 ms), hop 128, two-sided (zero Doppler centred), power in dB,
normalized per spectrogram to its max, clipped at -60 dB and scaled to [0, 1]; resized to
64 (Doppler) x 64 (time) by block averaging.
"""
import numpy as np
from scipy.signal import stft

from src.radar.params import ClassDwell
from src.radar.targets import bird, drone, slow_time

N_SLICES = 3
SPEC = 64


def spectrogram(x, prf):
    f, t, Z = stft(x, fs=prf, window="hann", nperseg=256, noverlap=128, return_onesided=False, boundary=None)
    P = np.fft.fftshift(np.abs(Z) ** 2, axes=0)
    db = 10 * np.log10(P + 1e-12)
    db = np.clip(db - db.max(), -60, 0) / 60 + 1
    F, T = db.shape
    db = db[: F - F % SPEC, : T - T % SPEC] if T >= SPEC else db
    f_blk = db.shape[0] // SPEC
    db = db[: f_blk * SPEC].reshape(SPEC, f_blk, -1).mean(1)
    T = db.shape[1]
    t_idx = np.linspace(0, T, SPEC + 1).astype(int)
    db = np.stack([db[:, t_idx[i]:max(t_idx[i + 1], t_idx[i] + 1)].mean(1) for i in range(SPEC)], axis=1)
    return db.astype(np.float32)


def make_instance(iid, kind, rng, dwell=ClassDwell()):
    v = float(rng.choice([-1, 1]) * rng.uniform(1, 15))
    snr = float(rng.uniform(-5, 25))
    if kind == "drone":
        sc, meta = drone(rng, 0.0, v, n_blades=int(rng.choice([2, 3])), blade_amp=float(rng.uniform(0.08, 0.20)))
    else:
        sc, meta = bird(rng, 0.0, v, wing_amp=float(rng.uniform(0.15, 0.35)))
    long = ClassDwell(fc=dwell.fc, prf=dwell.prf, duration=dwell.duration * N_SLICES)
    x = slow_time(sc, long, rng, snr)
    n = dwell.n_pulses
    specs = [spectrogram(x[i * n:(i + 1) * n], dwell.prf) for i in range(N_SLICES)]
    meta.update({"id": iid, "snr_db": snr, "v": v, "label": 1 if kind == "drone" else 0})
    return specs, meta


def make_dataset(n_per_class, seed):
    rng = np.random.default_rng(seed)
    X, y, ids, metas = [], [], [], []
    kinds = ["drone"] * n_per_class + ["bird"] * n_per_class
    for iid, kind in enumerate(kinds):
        specs, meta = make_instance(iid, kind, rng)
        for s in specs:
            X.append(s)
            y.append(meta["label"])
            ids.append(iid)
        metas.append(meta)
    return np.stack(X), np.array(y), np.array(ids), metas


def split_by_instance(ids, metas, seed, frac=(0.6, 0.2, 0.2)):
    """Leakage-safe: whole instances go to exactly one split (stratified by class)."""
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for lab in (0, 1):
        inst = np.array([m["id"] for m in metas if m["label"] == lab])
        rng.shuffle(inst)
        a, b = int(frac[0] * len(inst)), int((frac[0] + frac[1]) * len(inst))
        out["train"] += inst[:a].tolist()
        out["val"] += inst[a:b].tolist()
        out["test"] += inst[b:].tolist()
    sets = {k: set(v) for k, v in out.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"])
    return {k: np.isin(ids, list(v)) for k, v in sets.items()}


def split_heldout_config(ids, metas):
    """Held-out configuration: train/val on drones with rotor rate < 100 rev/s and birds with
    flap rate < 8 Hz; test on drones 100-120 rev/s and birds 8-10 Hz (unseen ranges)."""
    held = {m["id"] for m in metas if (m["kind"] == "drone" and m["f_rot"] >= 100) or
            (m["kind"] == "bird" and m["f_flap"] >= 8)}
    seen = [m for m in metas if m["id"] not in held]
    rng = np.random.default_rng(0)
    seen_ids = np.array([m["id"] for m in seen])
    rng.shuffle(seen_ids)
    nv = len(seen_ids) // 5
    val, train = set(seen_ids[:nv].tolist()), set(seen_ids[nv:].tolist())
    assert not (train & held) and not (val & held) and not (train & val)
    return {"train": np.isin(ids, list(train)), "val": np.isin(ids, list(val)), "test": np.isin(ids, list(held))}


def split_random_slices(n, seed, frac=(0.6, 0.2, 0.2)):
    """DELIBERATELY LEAKY: slices assigned at random, so one instance's slices land in train and test."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    m = {k: np.zeros(n, bool) for k in ("train", "val", "test")}
    m["train"][p[:a]] = True
    m["val"][p[a:b]] = True
    m["test"][p[b:]] = True
    return m

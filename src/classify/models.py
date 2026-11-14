"""Classifiers for 64x64 micro-Doppler spectrograms.

TinyCNN (< 20k params): conv(1->8,3) - pool - conv(8->16,3) - pool - conv(16->24,3) - pool -
global average - linear(24->2). Trained with Adam, early stopping on the validation split.

Feature baseline: handcrafted physics features -> standardized logistic regression:
  Doppler spread (power-weighted std of the Doppler profile), occupied bandwidth above -30 dB,
  spectral entropy of the time-averaged Doppler profile, peak-to-mean ratio, temporal
  periodicity (max normalized autocorrelation of the high-Doppler energy over time lags),
  energy fraction outside the central (body) band.

INT8: post-training quantization SIMULATED with fake quantization: per-output-channel
symmetric int8 weights, per-tensor symmetric int8 activations calibrated on training data
(max |activation| per layer). This estimates INT8 accuracy; it is not an integer kernel.
"""
import numpy as np
import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.c3 = nn.Conv2d(16, 24, 3, padding=1)
        self.fc = nn.Linear(24, 2)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, q=None):
        acts = [self.c1, self.c2, self.c3]
        for i, c in enumerate(acts):
            x = self.pool(torch.relu(c(x)))
            if q is not None:
                x = q(i, x)
        x = x.mean(dim=(2, 3))
        return self.fc(x)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def train_cnn(Xtr, ytr, Xva, yva, seed, epochs=40, batch=64, lr=3e-3, patience=6):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = TinyCNN()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt, yt = torch.from_numpy(Xtr[:, None]), torch.from_numpy(ytr).long()
    Xv, yv = torch.from_numpy(Xva[:, None]), torch.from_numpy(yva).long()
    g = torch.Generator().manual_seed(seed)
    best, state, bad = np.inf, None, 0
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(Xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(m(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(lossf(m(Xv), yv))
        if vl < best - 1e-4:
            best, bad, state = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(state)
    m.eval()
    return m


def predict(m, X, q=None):
    with torch.no_grad():
        return m(torch.from_numpy(X[:, None]), q=q).argmax(1).numpy()


def int8_model(m, Xcal):
    """Fake-quantized copy: int8 per-channel weights, int8 per-tensor activations."""
    import copy
    mq = copy.deepcopy(m)
    with torch.no_grad():
        for layer in (mq.c1, mq.c2, mq.c3, mq.fc):
            w = layer.weight
            s = w.abs().reshape(w.shape[0], -1).max(1).values.clamp(min=1e-12) / 127
            shape = (-1,) + (1,) * (w.dim() - 1)
            layer.weight.copy_(torch.round(w / s.reshape(shape)).clamp(-127, 127) * s.reshape(shape))
        # activation ranges from calibration data (inputs are in [0,1]: 8-bit grid of 1/255)
        ranges = [0.0, 0.0, 0.0]

        def rec(i, x):
            ranges[i] = max(ranges[i], float(x.abs().max()))
            return x
        mq(torch.from_numpy(Xcal[:, None]), q=rec)
    scales = [max(r, 1e-8) / 127 for r in ranges]

    def q(i, x):
        return torch.round(x / scales[i]).clamp(-127, 127) * scales[i]

    def predict_q(X):
        Xq = np.round(X * 255) / 255
        return predict(mq, Xq.astype(np.float32), q=q)
    return predict_q


def features(X):
    """Handcrafted micro-Doppler features, one row per spectrogram (64 Doppler x 64 time)."""
    P = 10 ** ((X * 60 - 60) / 10)                     # back to linear power (relative)
    prof = P.mean(axis=2)                               # Doppler profile
    prof = prof / prof.sum(1, keepdims=True)
    f = np.arange(X.shape[1]) - X.shape[1] / 2
    mu = (prof * f).sum(1)
    spread = np.sqrt((prof * (f - mu[:, None]) ** 2).sum(1))
    db = 10 * np.log10(prof / prof.max(1, keepdims=True) + 1e-12)
    occ = (db > -30).sum(1)
    ent = -(prof * np.log(prof + 1e-12)).sum(1)
    ptm = prof.max(1) / prof.mean(1)
    centre = np.abs(f - mu[:, None]) <= 3
    outside = 1 - (prof * centre).sum(1)
    hi = (P * (np.abs(f) > 6)[None, :, None]).sum(1)     # high-Doppler energy vs time
    hi = hi - hi.mean(1, keepdims=True)
    ac = np.array([np.max(np.correlate(h, h, "full")[len(h) + 2:] / (np.dot(h, h) + 1e-12)) for h in hi])
    return np.column_stack([spread, occ, ent, ptm, outside, ac])

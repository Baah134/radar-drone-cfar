"""Drone and bird scatterer models (shared by detection scenes and the micro-Doppler dwell).

Ranges are radial distances; the radar sees only the radial (line-of-sight) component of
motion. A target is a list of point scatterers, each with r(t) (m) and a complex amplitude
per pulse. Amplitudes are relative to the body (body amplitude = 1); scenes scale them.

Drone (quadcopter): body + 4 rotors x n_blades; each blade is modelled by N_PTS points
along the blade (root to tip), rotating at rate f_rot; the radial offset of a point at
radius rho is rho * cos(theta) * cos(elev) (rotor plane seen at aspect elevation 'elev').
Blade RCS is small: per-point amplitude blade_amp (relative to body).
Bird: body (Swerling-like slow RCS fluctuation) + 2 wings flapping at f_flap; each wing is
N_PTS points whose radial offset oscillates with amplitude ~ point radius * sin(flap angle).
"""
import numpy as np

N_PTS = 4


def drone(rng, r0, v, *, f_rot=None, n_blades=2, blade_len=None, elev=None, blade_amp=0.06,
          rotor_arm=0.12):
    f_rot = rng.uniform(50, 120) if f_rot is None else f_rot
    L = rng.uniform(0.10, 0.15) if blade_len is None else blade_len
    elev = rng.uniform(np.deg2rad(5), np.deg2rad(40)) if elev is None else elev
    sc = [{"r_of_t": (lambda t, r0=r0, v=v: r0 + v * t), "amp": 1.0}]
    for k in range(4):
        arm = rotor_arm * np.cos(k * np.pi / 2 + rng.uniform(-0.2, 0.2))      # rotor hub radial offset
        fr = f_rot * rng.uniform(0.95, 1.05) * (1 if k % 2 == 0 else -1)       # CW/CCW pairs
        ph0 = rng.uniform(0, 2 * np.pi)
        for b in range(n_blades):
            for p in range(N_PTS):
                rho = L * (p + 1) / N_PTS
                ph = ph0 + 2 * np.pi * b / n_blades
                sc.append({"r_of_t": (lambda t, r0=r0, v=v, a=arm, rho=rho, ph=ph, fr=fr:
                                      r0 + v * t + a + rho * np.cos(2 * np.pi * fr * t + ph) * np.cos(elev)),
                           "amp": blade_amp * (p + 1) / N_PTS})
    meta = {"kind": "drone", "f_rot": float(f_rot), "n_blades": int(n_blades), "blade_len": float(L),
            "elev_deg": float(np.rad2deg(elev)), "tip_speed": float(2 * np.pi * f_rot * L)}
    return sc, meta


def bird(rng, r0, v, *, f_flap=None, span=None, wing_amp=0.25, fluct_hz=None):
    f_flap = rng.uniform(2, 10) if f_flap is None else f_flap
    span = rng.uniform(0.3, 1.0) if span is None else span
    fluct_hz = rng.uniform(0.5, 3) if fluct_hz is None else fluct_hz
    ph_f = rng.uniform(0, 2 * np.pi)
    a1, a2 = rng.normal(size=2)
    body_amp = (lambda t, a1=a1, a2=a2: np.sqrt(np.abs(1 + 0.5 * (a1 * np.cos(2 * np.pi * fluct_hz * t)
                                                                  + a2 * np.sin(2 * np.pi * fluct_hz * t)))))
    sc = [{"r_of_t": (lambda t, r0=r0, v=v: r0 + v * t), "amp_t": body_amp, "amp": 1.0}]
    flap_ang = np.deg2rad(rng.uniform(30, 60))
    aspect = rng.uniform(0.2, 1.0)                 # how much flap motion projects onto the line of sight
    for side in (-1, 1):
        for p in range(N_PTS):
            rho = span / 2 * (p + 1) / N_PTS
            sc.append({"r_of_t": (lambda t, r0=r0, v=v, rho=rho:
                                  r0 + v * t + aspect * rho * np.sin(flap_ang * np.sin(2 * np.pi * f_flap * t + ph_f))),
                       "amp": wing_amp * (p + 1) / N_PTS})
    meta = {"kind": "bird", "f_flap": float(f_flap), "span": float(span), "fluct_hz": float(fluct_hz),
            "aspect": float(aspect)}
    return sc, meta


def to_frame_scatterers(sc, scale, Tc):
    """Convert to src.radar.sim.beat_signal format (per-chirp amplitude callables)."""
    out = []
    for s in sc:
        amp_t = s.get("amp_t")
        a = s["amp"] * scale
        out.append({"r_of_t": s["r_of_t"],
                    "amp": (lambda m, a=a, f=amp_t: a * (f(m * Tc) if f is not None else np.ones(len(m))))})
    return out


def slow_time(sc, dwell, rng, snr_db, scale_body=1.0):
    """Classification dwell at the target range bin: x[m] = sum a_i exp(j 4 pi r_i(t_m)/lam) + noise.
    snr_db: body SNR per pulse (unit noise variance)."""
    t = np.arange(dwell.n_pulses) / dwell.prf
    a0 = np.sqrt(10 ** (snr_db / 10)) * scale_body
    x = np.zeros(t.size, complex)
    for s in sc:
        amp = s["amp"] * a0 * (s["amp_t"](t) if "amp_t" in s else 1.0)
        x += amp * np.exp(1j * 4 * np.pi * s["r_of_t"](t) / dwell.lam)
    x += (rng.standard_normal(t.size) + 1j * rng.standard_normal(t.size)) / np.sqrt(2)
    return x

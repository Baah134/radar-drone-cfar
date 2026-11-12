"""FMCW radar parameters and derived quantities.

Detection waveform (spec defaults): 24 GHz, 250 MHz sweep, 100 us chirps, 256 complex
(I/Q) samples per chirp, 128 chirps per frame, chirps back to back (repetition = Tc).

Classification dwell (micro-Doppler, spec option (a)): a separate high-PRF dwell whose
unambiguous velocity covers rotor blade tips (up to ~2*pi*0.15 m*120 rev/s = 113 m/s).
Only the slow-time signal at the target's range bin is needed for the spectrogram, so
the dwell is simulated directly in slow time (one complex sample per chirp).
"""
from dataclasses import dataclass

import numpy as np

C = 299_792_458.0


@dataclass(frozen=True)
class FMCW:
    fc: float = 24e9          # carrier (Hz)
    B: float = 250e6          # sweep bandwidth (Hz)
    Tc: float = 100e-6        # chirp duration = chirp repetition interval (s)
    Ns: int = 256             # complex samples per chirp
    Nc: int = 128             # chirps per frame

    @property
    def lam(self):
        return C / self.fc

    @property
    def slope(self):
        return self.B / self.Tc

    @property
    def fs(self):
        """complex ADC rate: Ns samples across the chirp"""
        return self.Ns / self.Tc

    @property
    def range_res(self):
        return C / (2 * self.B)

    @property
    def range_max(self):
        """complex sampling: beat frequencies 0..fs map to ranges 0..R_max"""
        return self.fs * C / (2 * self.slope)

    @property
    def v_max(self):
        return self.lam / (4 * self.Tc)

    @property
    def v_res(self):
        return self.lam / (2 * self.Nc * self.Tc)

    @property
    def frame_time(self):
        return self.Nc * self.Tc

    def derived(self):
        return {"wavelength_m": self.lam, "slope_Hz_per_s": self.slope, "adc_rate_complex_Hz": self.fs,
                "range_resolution_m": self.range_res, "max_range_m": self.range_max,
                "range_bins": self.Ns, "max_unambiguous_velocity_mps": self.v_max,
                "velocity_resolution_mps": self.v_res, "doppler_bins": self.Nc,
                "frame_time_s": self.frame_time}


@dataclass(frozen=True)
class ClassDwell:
    """Micro-Doppler classification dwell: short chirps, high PRF."""
    fc: float = 24e9
    prf: float = 50e3          # chirp repetition frequency (Hz) -> v_max = lam*prf/4
    duration: float = 0.4      # s (covers >= 0.8 flap periods at 2 Hz, 4 at 10 Hz)

    @property
    def lam(self):
        return C / self.fc

    @property
    def v_max(self):
        return self.lam * self.prf / 4

    @property
    def n_pulses(self):
        return int(round(self.prf * self.duration))

    def derived(self):
        return {"prf_Hz": self.prf, "max_unambiguous_velocity_mps": self.v_max, "duration_s": self.duration,
                "pulses": self.n_pulses, "doppler_span_Hz": self.prf,
                "max_blade_tip_speed_mps_in_dataset": float(2 * np.pi * 0.15 * 120)}

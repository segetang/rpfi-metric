"""Synthetic reference PPG generators: synth_ppg (degradation batteries) and Pulse (Experiment 1, synthetic mode).

Copied unchanged (code) from the v1 repository's derive_anchors.py; comments translated.
"""
import numpy as np


def synth_ppg(n, fs, rng, hr_bpm=None):
    """
    Physiologically plausible pulse wave: fundamental + harmonics (dicrotic notch) +
    respiratory sinus arrhythmia (RSA) + slow amplitude modulation.
    """
    if hr_bpm is None:
        hr_bpm = rng.uniform(55, 100)
    t = np.arange(n) / fs
    f0 = hr_bpm / 60.0

    # RSA: instantaneous frequency modulated by respiration (0.2-0.35 Hz) -> integrate phase
    f_resp = rng.uniform(0.2, 0.35)
    inst_f = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * f_resp * t + rng.uniform(0, 2 * np.pi)))
    phase = 2 * np.pi * np.cumsum(inst_f) / fs

    sig = (np.sin(phase)
           + 0.35 * np.sin(2 * phase + 0.8)      # dicrotic notch
           + 0.12 * np.sin(3 * phase + 1.6))
    sig *= (1.0 + 0.10 * np.sin(2 * np.pi * f_resp * t))   # amplitude modulation
    return sig, hr_bpm


class Pulse:
    """Continuous-time synthetic PPG (used by exp1_distortion_id.py in synthetic mode).
    The phase is integrated analytically, so it can be evaluated exactly at any time and lag.
    Copied unchanged (code) from the analysis-time rhythm_diagnostics.py."""

    def __init__(self, rng, hr=None, rsa=0.06):
        self.hr = rng.uniform(55, 100) if hr is None else hr
        self.f0 = self.hr / 60
        self.fr = rng.uniform(0.2, 0.35)
        self.ph0 = rng.uniform(0, 2 * np.pi)
        self.phr = rng.uniform(0, 2 * np.pi)
        self.rsa = rsa

    def with_(self, **kw):
        p = Pulse.__new__(Pulse); p.__dict__ = {**self.__dict__, **kw}; return p

    def phase(self, t):
        a, fr = self.rsa, self.fr
        return 2 * np.pi * self.f0 * (t - a / (2 * np.pi * fr) * (np.cos(2 * np.pi * fr * t + self.phr) - np.cos(self.phr))) + self.ph0

    def __call__(self, t):
        p = self.phase(t)
        s = np.sin(p) + 0.35 * np.sin(2 * p + 0.8) + 0.12 * np.sin(3 * p + 1.6)
        return s * (1 + 0.10 * np.sin(2 * np.pi * self.fr * t))

    def cycle_starts(self, t):
        k = np.floor((self.phase(t) - self.ph0) / (2 * np.pi))
        return t[1:][np.diff(k) > 0]

"""Data-free regression tests (pytest). The profiles of 14 synthetic pulse cases must match the stored values
(tolerance 1e-4 absorbs floating-point differences across platforms).
Run from the repository root:  python -m pytest tests -q"""
import json, sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rpfi"))
import rpfi_v2 as RV

EXP = json.load(open(Path(__file__).with_name("expected_v2.0.2.json"), encoding="utf-8"))


def ppg(t, hr=72.0):
    x = (t * hr / 60.0) % 1.0
    return np.exp(-((x - 0.18) / 0.07) ** 2) + 0.45 * np.exp(-((x - 0.45) / 0.10) ** 2)


def make(fs):
    t = np.arange(60 * fs) / fs; y = ppg(t); rng = np.random.default_rng(123)
    return y, {"identical": y, "scaled_offset": 3 * y + 5, "lag_100ms": ppg(t - 0.1),
               "noise_1sd": y + y.std() * rng.standard_normal(t.size), "inverted": -y,
               "unrelated_95bpm": ppg(t, 95.0), "flat": np.zeros_like(y)}


@pytest.mark.parametrize("key", sorted(EXP))
def test_profile_matches_reference(key):
    fs = int(key.split("_")[0][2:]); name = key.split("_", 1)[1]
    y, preds = make(fs); r = RV.score(y, preds[name], fs)
    for k, v in EXP[key].items():
        if isinstance(v, bool):
            assert bool(r[k]) == v, (key, k)
        else:
            assert r[k] == pytest.approx(v, abs=1e-4), (key, k)   # OS/BLAS differences of order 1e-6 (Windows vs. Linux measured <= 6e-6)


def test_properties():
    y, p = make(30)
    assert RV.score(y, p["identical"], 30)["RPFI"] == pytest.approx(1.0)
    assert RV.score(y, p["scaled_offset"], 30)["RPFI"] == pytest.approx(1.0, abs=1e-4)       # amplitude/offset invariance
    assert RV.score(y, p["flat"], 30)["RPFI"] == 0.0                                 # flat output = 0
    assert RV.score(y, p["unrelated_95bpm"], 30)["SHAPE_defined"] is False          # unrelated-signal gate
    assert RV.score(y, p["inverted"], 30)["polarity_suspect"] is True               # polarity flag
    assert abs(RV.score(y, p["lag_100ms"], 30)["lag_s"] - 0.1) < 0.01                # lag separated

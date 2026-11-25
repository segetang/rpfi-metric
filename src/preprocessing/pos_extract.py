"""
pos_extract.py -- reference implementation of the POS (Plane-Orthogonal-to-Skin) rPPG extractor.
Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 64(7):1479-1491, 2017.

Provenance
---------
The extractor traces analysed in the paper are a .npy archive extracted beforehand with the remotebiosensing/rppg
(https://github.com/remotebiosensing/rppg) pipeline. This file is not that original extraction script but a
self-written reference implementation of the published algorithm, so that the extraction step can be reproduced.
ROI/skin-segmentation settings differ between pipelines, so traces re-extracted with this code are not
bit-identical to the archived arrays.

For the front end (video -> face ROI -> mean RGB) see roi_extract.py.

Usage
------
    from roi_extract import extract_roi_video, rgb_series
    from pos_extract import pos

    frames = extract_roi_video('subject.avi', img_size=128)   # (T, S, S, 3)
    rgb    = rgb_series(frames)                               # (T, 3)
    bvp    = pos(rgb, fs=30)                                  # (T,)
"""
import numpy as np

# Projection matrix of Wang et al. (2017), Eq. (10).
# Its two rows span the plane orthogonal to the normalized skin-tone vector.
P_POS = np.array([[0.0, 1.0, -1.0],
                  [-2.0, 1.0, 1.0]])


def pos(rgb, fs=30, window_seconds=1.6):
    """Extract the POS pulse signal from an RGB time series.

    Args:
        rgb: (T, 3) per-frame ROI mean RGB.
        fs: frame rate (Hz).
        window_seconds: sliding-window length. 1.6 s by default, as in the original paper.

    Returns:
        (T,) pulse signal (overlap-add, zero mean).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f'rgb must have shape (T, 3); got {rgb.shape}')
    T = rgb.shape[0]
    l = int(round(window_seconds * fs))          # window length (frames)
    if T < l:
        raise ValueError(f'signal length ({T}) is shorter than the window ({l})')
    H = np.zeros(T, dtype=np.float64)

    for n in range(l - 1, T):
        m = n - l + 1
        block = rgb[m:n + 1]                     # (l, 3)

        # (1) temporal normalization: divide each channel by its window mean
        mu = block.mean(axis=0)
        mu[mu == 0] = 1e-12
        Cn = block / mu

        # (2) projection onto the plane orthogonal to the skin tone
        S = P_POS @ Cn.T                         # (2, l)

        # (3) alpha tuning: combine the two projected components
        s1, s2 = S[0], S[1]
        std2 = s2.std()
        alpha = (s1.std() / std2) if std2 > 1e-12 else 0.0
        h = s1 + alpha * s2

        # (4) overlap-add of the zero-mean window
        H[m:n + 1] += (h - h.mean())

    return H


if __name__ == '__main__':
    # self-check with a synthetic pulse plus illumination drift
    fs, dur, hr = 30, 20, 1.2                       # 1.2 Hz = 72 BPM
    t = np.arange(fs * dur) / fs
    pulse = np.sin(2 * np.pi * hr * t)
    base = np.array([180.0, 120.0, 100.0])
    gain = np.array([0.6, 1.0, 0.4])                # green dominant, as in real skin
    rng = np.random.default_rng(0)
    rgb = base + np.outer(pulse, base * gain * 0.01) + rng.normal(0, 0.05, (len(t), 3))
    rgb += np.outer(np.linspace(0, 6, len(t)), np.ones(3))   # illumination drift

    bvp = pos(rgb, fs=fs)
    f = np.fft.rfftfreq(len(bvp), 1 / fs)
    peak = f[np.argmax(np.abs(np.fft.rfft(bvp - bvp.mean()))[1:]) + 1]
    print(f'self-check: recovered {peak:.2f} Hz ({peak * 60:.0f} BPM), expected {hr:.2f} Hz')
    assert abs(peak - hr) < 0.1, 'POS self-check failed'
    print('POS self-check passed')

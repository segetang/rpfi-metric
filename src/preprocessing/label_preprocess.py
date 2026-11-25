"""
label_preprocess.py -- contact-PPG label preprocessing for the 3 datasets (PURE, UBFC-rPPG, COHFACE),
and a check of the actual state of the distributed .npy label arrays.

Key point: apart from segmentation and concatenation, no normalization is applied to the distributed label arrays.
The filter chain is split into three branches by purpose, each with different parameters.

  (A) diagnostics/plots  : detrend(lambda=100) -> Butterworth order 6, 0.75-3.0 Hz -> z-norm
                           (process_signal in rPPG/signal_utils.py)
  (B) alignment check    : Butterworth order 3, 0.7-3.5 Hz            (build_aligned_inputs.py)
  (C) training/evaluation: no band-pass; standardized with training-split statistics only
                           (train_fast.py, train_subjectcv.py)

(C) is the only processing the models actually see; predictions are inverse-transformed before scoring, so MAE/RMSE
are reported in the label's original scale (order of 10 for PURE/COHFACE).

Run:  python label_preprocess.py
"""
import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, filtfilt

PRED = '../rPPG/predictions'
SEG = 300                       # archive segment length (frames)
DATASETS = {                    # name: (fs, display name)
    'PURE': (30, 'PURE'),
    'UBFC': (30, 'UBFC-rPPG'),
    'cohface': (20, 'COHFACE'),
}


# --------------------------------------------------------------------------
# (A) diagnostic chain -- identical to rPPG/signal_utils.py
# --------------------------------------------------------------------------
def detrend(signal, Lambda=100):
    """Tarvainen et al. (2002) smoothness-prior detrending. lambda = 100 fixed."""
    n = len(signal)
    H = np.identity(n)
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    D = spdiags(np.array([ones, minus_twos, ones]),
                np.array([0, 1, 2]), (n - 2), n).toarray()
    return np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)


def bpf_diagnostic(x, fs=30, low=0.75, high=3.0, order=6):
    """Diagnostic band-pass: Butterworth order 6, 0.75-3.0 Hz, zero-phase."""
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='bandpass')
    return filtfilt(b, a, np.double(x))


def z_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def process_signal(signal, fs=30, detrend_on=True, bpf_on=True, Lambda=100):
    """Whole (A) chain: detrend -> BPF -> z-norm."""
    proc = np.asarray(signal, dtype=np.float64).copy()
    if detrend_on:
        proc = detrend(proc, Lambda)
    if bpf_on:
        proc = bpf_diagnostic(proc, fs=fs)
    return z_normalize(proc)


# --------------------------------------------------------------------------
# (B) alignment band-pass -- identical to build_aligned_inputs.py (different parameters)
# --------------------------------------------------------------------------
def bpf_alignment(x, fs, low=0.7, high=3.5, order=3):
    """Alignment band-pass: Butterworth order 3, 0.7-3.5 Hz. Used only for lag estimation."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


# --------------------------------------------------------------------------
# (C) training standardization -- training-split statistics only
# --------------------------------------------------------------------------
def standardize_train_only(x, train_end):
    """Standardize with the mean/std of the training split only (no val/test leakage)."""
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x[:train_end].mean(), x[:train_end].std() + 1e-8
    return (x - mu) / sd, mu, sd


def align_label_to_frames(label, frame_total):
    """If the label length differs from the number of frames, align it by np.interp interpolation.
    (same alignment rule as get_label() of the original extraction pipeline)"""
    label = np.asarray(label, dtype=np.float64)
    if len(label) == frame_total:
        return label
    return np.interp(np.linspace(0, len(label) - 1, frame_total),
                     np.arange(len(label)), label)


# --------------------------------------------------------------------------
# Check: actual state of the distributed label arrays
# --------------------------------------------------------------------------
def verify():
    print(f"{'dataset':<12}{'frames':>9}{'fs':>4}{'segments':>9}"
          f"{'global mean':>11}{'global std':>10}{'seg1 mean':>12}{'seg1 std':>11}")
    print('-' * 80)
    for ds, (fs, name) in DATASETS.items():
        lab = np.load(f'{PRED}/{ds}_label.npy').astype(np.float64).flatten()
        seg0 = lab[:SEG]
        print(f'{name:<12}{len(lab):>9d}{fs:>4d}{len(lab)//SEG:>9d}'
              f'{lab.mean():>11.3f}{lab.std():>10.3f}'
              f'{seg0.mean():>12.3f}{seg0.std():>11.3f}')
    print('\n[interpretation] global and first-segment statistics differ, so no per-segment normalization '
          'is applied.\n'
          '        Only UBFC-rPPG happens to be globally z-scored; PURE/COHFACE are in original units.')

    # example of the (A) chain
    lab = np.load(f'{PRED}/PURE_label.npy').astype(np.float64).flatten()[:SEG]
    proc = process_signal(lab, fs=30)
    print(f'\n(A) diagnostic chain example [PURE first segment]: '
          f'raw mean={lab.mean():.3f} std={lab.std():.3f}  ->  '
          f'processed mean={proc.mean():.3e} std={proc.std():.3f}')

    # example of the (C) training standardization
    std_x, mu, sd = standardize_train_only(lab, train_end=int(len(lab) * 0.7))
    print(f'(C) training standardization example: train split mu={mu:.3f}, sd={sd:.3f} '
          f'-> overall mean={std_x.mean():.3f} std={std_x.std():.3f}')

    # label alignment example
    short = np.sin(np.linspace(0, 4 * np.pi, 250))
    print(f'\nlabel alignment example: len(label)=250, frame_total=300 -> '
          f'{len(align_label_to_frames(short, 300))} (np.interp interpolation)')


if __name__ == '__main__':
    verify()

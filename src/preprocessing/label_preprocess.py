"""
label_preprocess.py -- 3개 데이터셋(PURE, UBFC-rPPG, COHFACE)의 contact-PPG 라벨
전처리 절차를 구현하고, 배포된 .npy 라벨 배열의 실제 상태를 검증한다.

핵심: 배포된 라벨 배열에는 분할·연결 외에 어떤 정규화도 적용되어 있지 않다.
필터 체인은 용도에 따라 세 갈래로 분리되어 있으며, 서로 다른 파라미터를 쓴다.

  (A) 상위 진단/플롯용   : detrend(λ=100) -> Butterworth 6차 0.75-3.0Hz -> z-norm
                           (rPPG/signal_utils.py의 process_signal)
  (B) 정렬(alignment) 진단: Butterworth 3차 0.7-3.5Hz            (build_aligned_inputs.py)
  (C) 학습/평가          : band-pass 없음. train 구간 통계로만 표준화
                           (train_fast.py, train_subjectcv.py)

(C)가 모델이 실제로 보는 유일한 처리이며, 예측은 역변환 후 채점하므로 MAE/RMSE는
라벨의 원 스케일(PURE/COHFACE 기준 10의 자리)로 보고된다.

실행:  python label_preprocess.py
"""
import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, filtfilt

PRED = '../rPPG/predictions'
SEG = 300                       # 아카이브 세그먼트 길이(프레임)
DATASETS = {                    # 이름: (fs, 표시명)
    'PURE': (30, 'PURE'),
    'UBFC': (30, 'UBFC-rPPG'),
    'cohface': (20, 'COHFACE'),
}


# --------------------------------------------------------------------------
# (A) 상위 진단용 체인 -- rPPG/signal_utils.py와 동일
# --------------------------------------------------------------------------
def detrend(signal, Lambda=100):
    """Tarvainen et al. (2002) smoothness-prior detrending. λ=100 고정."""
    n = len(signal)
    H = np.identity(n)
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    D = spdiags(np.array([ones, minus_twos, ones]),
                np.array([0, 1, 2]), (n - 2), n).toarray()
    return np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)


def bpf_diagnostic(x, fs=30, low=0.75, high=3.0, order=6):
    """상위 진단용 band-pass: Butterworth 6차, 0.75-3.0 Hz, zero-phase."""
    b, a = butter(order, [low / (0.5 * fs), high / (0.5 * fs)], btype='bandpass')
    return filtfilt(b, a, np.double(x))


def z_normalize(x):
    return (x - np.mean(x)) / np.std(x)


def process_signal(signal, fs=30, detrend_on=True, bpf_on=True, Lambda=100):
    """(A) 체인 전체: detrend -> BPF -> z-norm."""
    proc = np.asarray(signal, dtype=np.float64).copy()
    if detrend_on:
        proc = detrend(proc, Lambda)
    if bpf_on:
        proc = bpf_diagnostic(proc, fs=fs)
    return z_normalize(proc)


# --------------------------------------------------------------------------
# (B) 정렬 진단용 band-pass -- build_aligned_inputs.py와 동일 (파라미터가 다름)
# --------------------------------------------------------------------------
def bpf_alignment(x, fs, low=0.7, high=3.5, order=3):
    """정렬용 band-pass: Butterworth 3차, 0.7-3.5 Hz. lag 추정에만 사용."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype='band')
    return filtfilt(b, a, x)


# --------------------------------------------------------------------------
# (C) 학습용 표준화 -- train 구간 통계만 사용
# --------------------------------------------------------------------------
def standardize_train_only(x, train_end):
    """train 구간의 mean/std로만 표준화한다 (val/test 누수 없음)."""
    x = np.asarray(x, dtype=np.float64)
    mu, sd = x[:train_end].mean(), x[:train_end].std() + 1e-8
    return (x - mu) / sd, mu, sd


def align_label_to_frames(label, frame_total):
    """라벨 길이가 프레임 수와 다르면 np.interp로 보간 정렬한다.
    (원 추출 파이프라인 get_label()의 정렬 규칙과 동일)"""
    label = np.asarray(label, dtype=np.float64)
    if len(label) == frame_total:
        return label
    return np.interp(np.linspace(0, len(label) - 1, frame_total),
                     np.arange(len(label)), label)


# --------------------------------------------------------------------------
# 검증: 배포된 라벨 배열의 실제 상태
# --------------------------------------------------------------------------
def verify():
    print(f"{'데이터셋':<12}{'프레임':>9}{'fs':>4}{'세그먼트':>9}"
          f"{'전역 mean':>11}{'전역 std':>10}{'첫세그 mean':>12}{'첫세그 std':>11}")
    print('-' * 80)
    for ds, (fs, name) in DATASETS.items():
        lab = np.load(f'{PRED}/{ds}_label.npy').astype(np.float64).flatten()
        seg0 = lab[:SEG]
        print(f'{name:<12}{len(lab):>9d}{fs:>4d}{len(lab)//SEG:>9d}'
              f'{lab.mean():>11.3f}{lab.std():>10.3f}'
              f'{seg0.mean():>12.3f}{seg0.std():>11.3f}')
    print('\n[해석] 전역 통계와 첫 세그먼트 통계가 다르므로 세그먼트별 정규화는 '
          '적용되어 있지 않다.\n'
          '        UBFC-rPPG만 우연히 전역 z-score 상태이고 PURE/COHFACE는 원 단위다.')

    # (A) 체인 동작 예시
    lab = np.load(f'{PRED}/PURE_label.npy').astype(np.float64).flatten()[:SEG]
    proc = process_signal(lab, fs=30)
    print(f'\n(A) 진단 체인 예시 [PURE 첫 세그먼트]: '
          f'원본 mean={lab.mean():.3f} std={lab.std():.3f}  ->  '
          f'처리 후 mean={proc.mean():.3e} std={proc.std():.3f}')

    # (C) 학습용 표준화 예시
    std_x, mu, sd = standardize_train_only(lab, train_end=int(len(lab) * 0.7))
    print(f'(C) 학습 표준화 예시: train 구간 mu={mu:.3f}, sd={sd:.3f} '
          f'-> 전체 mean={std_x.mean():.3f} std={std_x.std():.3f}')

    # 라벨 정렬 예시
    short = np.sin(np.linspace(0, 4 * np.pi, 250))
    print(f'\n라벨 정렬 예시: len(label)=250, frame_total=300 -> '
          f'{len(align_label_to_frames(short, 300))} (np.interp 보간)')


if __name__ == '__main__':
    verify()

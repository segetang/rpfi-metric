"""
pos_extract.py -- POS(Plane-Orthogonal-to-Skin) rPPG 추출기 참조 구현.
Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 64(7):1479-1491, 2017.

출처 안내
---------
논문에서 분석한 extractor trace는 remotebiosensing/rppg
(https://github.com/remotebiosensing/rppg) 파이프라인으로 미리 추출해 둔 .npy
아카이브다. 본 파일은 그 원본 추출 스크립트가 아니라, 추출 단계를 재현할 수 있도록
공개 알고리즘을 자체 구현한 참조 코드다. ROI/스킨 분할 설정이 파이프라인마다
다르므로 이 코드로 다시 뽑은 trace는 아카이브 배열과 비트 단위로 일치하지 않는다.

앞단(비디오 -> 얼굴 ROI -> 평균 RGB)은 roi_extract.py를 참고할 것.

사용법
------
    from roi_extract import extract_roi_video, rgb_series
    from pos_extract import pos

    frames = extract_roi_video('subject.avi', img_size=128)   # (T, S, S, 3)
    rgb    = rgb_series(frames)                               # (T, 3)
    bvp    = pos(rgb, fs=30)                                  # (T,)
"""
import numpy as np

# Wang et al.(2017) 식 (10)의 투영 행렬.
# 표준화된 피부톤 벡터에 직교하는 평면을 두 행이 span한다.
P_POS = np.array([[0.0, 1.0, -1.0],
                  [-2.0, 1.0, 1.0]])


def pos(rgb, fs=30, window_seconds=1.6):
    """RGB 시계열에서 POS 맥파를 추출한다.

    Args:
        rgb: (T, 3) 프레임별 ROI 평균 RGB.
        fs: 프레임률 (Hz).
        window_seconds: 슬라이딩 윈도 길이. 원 논문 기본값 1.6초.

    Returns:
        (T,) 맥파 신호 (overlap-add, 평균 0).
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f'rgb는 (T, 3) 형태여야 한다. 입력: {rgb.shape}')
    T = rgb.shape[0]
    l = int(round(window_seconds * fs))          # 윈도 길이(프레임)
    if T < l:
        raise ValueError(f'신호 길이({T})가 윈도({l})보다 짧다')
    H = np.zeros(T, dtype=np.float64)

    for n in range(l - 1, T):
        m = n - l + 1
        block = rgb[m:n + 1]                     # (l, 3)

        # (1) 시간 정규화: 각 채널을 윈도 평균으로 나눔
        mu = block.mean(axis=0)
        mu[mu == 0] = 1e-12
        Cn = block / mu

        # (2) 피부톤 직교 평면으로 투영
        S = P_POS @ Cn.T                         # (2, l)

        # (3) alpha 튜닝: 두 투영 성분 결합
        s1, s2 = S[0], S[1]
        std2 = s2.std()
        alpha = (s1.std() / std2) if std2 > 1e-12 else 0.0
        h = s1 + alpha * s2

        # (4) 평균 0으로 만든 윈도를 overlap-add
        H[m:n + 1] += (h - h.mean())

    return H


if __name__ == '__main__':
    # 조명 드리프트를 넣은 합성 맥파로 자체 점검
    fs, dur, hr = 30, 20, 1.2                       # 1.2 Hz = 72 BPM
    t = np.arange(fs * dur) / fs
    pulse = np.sin(2 * np.pi * hr * t)
    base = np.array([180.0, 120.0, 100.0])
    gain = np.array([0.6, 1.0, 0.4])                # 실제 피부와 비슷하게 green 우세
    rng = np.random.default_rng(0)
    rgb = base + np.outer(pulse, base * gain * 0.01) + rng.normal(0, 0.05, (len(t), 3))
    rgb += np.outer(np.linspace(0, 6, len(t)), np.ones(3))   # 조명 드리프트

    bvp = pos(rgb, fs=fs)
    f = np.fft.rfftfreq(len(bvp), 1 / fs)
    peak = f[np.argmax(np.abs(np.fft.rfft(bvp - bvp.mean()))[1:]) + 1]
    print(f'자체 점검: 복원 {peak:.2f} Hz ({peak * 60:.0f} BPM), 기대값 {hr:.2f} Hz')
    assert abs(peak - hr) < 0.1, 'POS 자체 점검 실패'
    print('POS 자체 점검 통과')

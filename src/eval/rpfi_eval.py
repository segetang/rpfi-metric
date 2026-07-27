"""
rpfi_eval.py — RPFI (rPPG Physiological Fidelity Index) 평가 파이프라인 (재작성판)
====================================================================================
main_eval_final.py를 대체한다. RPFI의 핵심 아이디어(BWMD 형태학 / SRE SNR가중오차 /
WCR 일치도 / EDD 오차분포 4개 컴포넌트를 합쳐 0-100 충실도 점수를 낸다)는 그대로
유지하되, 리뷰어 지적과 자체 코드감사에서 나온 결함을 전부 반영했다.

기존 main_eval_final.py 대비 변경점 (→ 대응 리뷰어 항목)
---------------------------------------------------------
[1] SRE saturation 항 부호 수정                                    → 자체발견 X-4
    기존:  g = 1 + w_c·cri·r_low + w_s·si·r_high   (고SNR일수록 페널티 ↑, 논문 서술과 반대)
           실측 g(30dB)=1.50, g(40dB)=1.83
    수정:  g = 1 + w_c·cri·r_low − w_s·si·r_high,  g ∈ [G_MIN, G_MAX]로 클램프
           g(30dB)=0.50, g(40dB)=0.20(하한)
    의미: 참조신호 SNR이 낮으면(어려운 조건) 같은 RMSE라도 정상참작 → 가중 ↑는
          부적절하므로 반대로, 고SNR(쉬운 조건)에서 난 오차는 변명의 여지가 없다는
          논문 서술에 맞춰 저SNR에서 가중을 키우고 고SNR에서 줄인다.

[2] percentile 정규화 → 고정 anchor                                → R1-2
    기존 percentile(5-95)는 "평가 대상 모델 풀"에 의존해서, 모델을 하나 빼고
    돌리면 남은 모델 점수가 바뀐다(= 절대적 충실도 지표가 아님).
    z-norm 평가경로에서 각 컴포넌트가 유계임을 이용해 고정 anchor로 교체.

[3] 비보상적(non-compensatory) 집계 추가                            → R1-2
    가중합(arithmetic)은 한 컴포넌트가 0이어도 나머지로 메워진다. 가중 기하평균을
    기본값으로 두어 "어느 한 축이 무너지면 전체 점수가 무너지게" 한다.
    두 방식 모두 출력하여 비교 가능.

[4] 분석 단위를 recording → participant로 교정                      → R1-7
    기존 subject_counts 하드코딩 리스트는 사실 recording 단위였고, 이것이
    Friedman χ² 상한 초과(PURE N=10인데 χ²=422.73)의 원인이었다.
    이제 runs/{ds}_cv{K}/*_info.json의 test_subject_of_range를 읽어
    participant 단위로 묶는다. K-fold 성질상 모든 participant가 정확히 1회씩 test에
    등장하므로, 전 fold를 모으면 참가자 전원이 정확히 한 번씩 집계된다.

[5] participant 단위 cluster bootstrap + 사후검정                   → R1-7
    bootstrap은 chunk가 아니라 participant를 리샘플링(클러스터 부트스트랩).
    Friedman이 유의하면 pairwise Wilcoxon signed-rank + Holm 보정으로 사후검정.

[6] 평가경로 전처리 일원화                                          → 자체발견 X-6
    기존에는 process_signal이 plot 함수에서만 호출되어, 채점은 원신호로 하고
    그림은 필터신호로 그리는 불일치가 있었다. 이제 label과 prediction에
    **완전히 동일한** 전처리(chunk → detrend → BPF → z-norm)를 적용한다.

[7] 컴포넌트 수준 근거 보고: HR MAE, PRV 표, 전면 CSV               → R1-6, R1-11
[8] MAPE 삭제, PSNR MAX 정의 명시                                   → R1-M1
[9] HRV → PRV 용어 교정, 단위 ms 통일                               → R1-M2, R1-5
    (기존 코드는 본문 "ms"/표 "seconds" 불일치 상태였다)
[10] 가중치 민감도 + 컴포넌트 중복성 분석                            → R1-11, R2
[11] 계산비용 벤치마크 유지                                          → R2-6

입력 형식 (train_models.py / train_bimamba_blend.py 산출물)
-------------------------------------------------------------
    runs/{dataset}_cv{K}/{model}_f{k}_test_pred.npy   object 배열, recording별 1D (z-space)
    runs/{dataset}_cv{K}/{model}_f{k}_info.json       test_ranges, test_subject_of_range 등
    ../real data/{DS}_label.npy                       데이터셋 전체 label (raw)

사용법
------
    python rpfi_eval.py --dataset ubfc \
        --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" \
        --out results_eval

    # 모델 일부만 / 대역통과 민감도 확인
    python rpfi_eval.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" \
        --models rnn,bilstm,bimamba --no-bpf --out results_eval_nobpf
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import spdiags
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon

try:
    from fastdtw import fastdtw as _fastdtw
    USE_FASTDTW = True
except ImportError:
    USE_FASTDTW = False

# numpy 2.0에서 np.trapz → np.trapezoid로 개명됨. 양쪽 버전 모두 지원.
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ══════════════════════════════════════════════════════════════════
# 설정 상수
# ══════════════════════════════════════════════════════════════════

FS_BY_DATASET = {"pure": 30, "ubfc": 30, "cohface": 20}

CARDIAC_BAND = (0.75, 3.0)          # Hz — 45~180 bpm
COMPONENTS = ["BWMD", "SRE", "WCR", "EDD"]

# 컴포넌트 방향: lower = 값이 작을수록 좋음
DIRECTION = {"BWMD": "lower", "SRE": "lower", "WCR": "higher", "EDD": "lower"}

# [2] 고정 anchor (풀 비의존).
#   WCR/EDD: 정의상 유계 — WCR은 [-1,1]이나 음의 상관은 전부 "최악"으로 보아 하한 0.
#            EDD는 정규화 JSD라 [0,1].
#   BWMD/SRE: z-norm 평가경로에서 이론 상한이 존재하나(각각 3.0, ~6.0) 실제 도달하지
#            않으므로, 합성 열화 벤치마크(rpfi_degradation.py) 분포의 p1-p99를 anchor로 쓴다.
#   ※ --anchors 로 JSON을 넘겨 덮어쓸 수 있고, 실행 시 실측 범위가 anchor를 벗어나면 경고한다.
DEFAULT_ANCHORS = {
    "BWMD": (0.009, 0.859),
    "SRE":  (0.065, 6.493),
    "WCR":  (0.0, 1.0),
    "EDD":  (0.0, 1.0),
}

DEFAULT_WEIGHTS = {"BWMD": 0.3, "SRE": 0.2, "WCR": 0.3, "EDD": 0.2}

# [1] SRE g(SNR) 파라미터
SRE_SNR_LOW = 3.0        # dB, 이 아래는 "어려운 캡처 조건"
SRE_SNR_HIGH = 15.0      # dB, 이 위는 "쉬운 캡처 조건"
SRE_W_LOW = 2.0          # 저SNR 가중 증폭 계수
SRE_W_HIGH = 0.5         # 고SNR 가중 감쇠 계수
# 유계성은 비율이 아니라 g 자체를 클램프해서 확보한다.
#   G_MIN: 아무리 쉬운 조건이라도 오차를 완전히 무시하지는 않는다는 하한
#   G_MAX: 극저SNR에서 페널티가 발산하지 않게 하는 상한
# 이 두 클램프로 SRE = RMSE·g 가 유계가 되어 고정 anchor가 성립한다.
# 결과: g(30dB)=0.50, g(40dB)=0.20(하한 도달), g(SNR→-∞)=3.00
SRE_G_MIN = 0.20
SRE_G_MAX = 3.00

BWMD_LAMBDA_P = 1.0      # peak timing 항 가중
BWMD_LAMBDA_A = 0.5      # amplitude overlap 항 가중
WCR_BETA = 0.6           # CCC vs Spearman 배합
# EDD_BAND는 CARDIAC_BAND를 그대로 재사용 (compute_edd 참조)


# ══════════════════════════════════════════════════════════════════
# 전처리 — label과 prediction에 완전히 동일하게 적용 (X-6)
# ══════════════════════════════════════════════════════════════════

def detrend_tarvainen(signal, Lambda=100):
    """Tarvainen et al. (2002) smoothness-priors detrending."""
    n = len(signal)
    if n < 3:
        return np.asarray(signal, dtype=float)
    H = np.identity(n)
    ones = np.ones(n)
    diags_data = np.array([ones, -2 * ones, ones])
    D = spdiags(diags_data, np.array([0, 1, 2]), (n - 2), n).toarray()
    return np.dot((H - np.linalg.inv(H + (Lambda ** 2) * np.dot(D.T, D))), signal)


def bandpass(x, fs, low=CARDIAC_BAND[0], high=CARDIAC_BAND[1], order=6):
    nyq = 0.5 * fs
    hi = min(high / nyq, 0.99)
    lo = max(low / nyq, 1e-6)
    if lo >= hi:
        return np.asarray(x, dtype=float)
    b, a = butter(order, [lo, hi], btype="bandpass")
    return filtfilt(b, a, np.asarray(x, dtype=np.double))


def znorm(x):
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    if s < 1e-12:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def preprocess_chunk(x, fs, do_detrend=True, do_bpf=True, Lambda=100):
    """chunk 단위 전처리. BWMD/SRE/WCR가 쓰는 표준 경로. label/pred 모두 동일 적용."""
    y = np.asarray(x, dtype=float).copy()
    if do_detrend and len(y) >= 3:
        y = detrend_tarvainen(y, Lambda)
    if do_bpf and len(y) > 18:      # filtfilt padlen 여유
        y = bandpass(y, fs)
    return znorm(y)


def preprocess_chunk_edd(x, fs, do_detrend=True, Lambda=100):
    """
    EDD 전용 전처리 — detrend + z-norm까지만, **BPF는 절대 적용하지 않는다**.

    이유: BWMD/SRE/WCR용 신호는 label/pred 모두 0.75-3.0Hz로 강제 필터링돼있어,
    그 신호로 잔차의 '심장대역 파워 비율'을 재면 분모(전체 파워)도 이미 그
    대역 안에 갇혀있으므로 비율이 항상 ~1.0으로 나와 아무것도 구분하지 못한다.
    EDD가 의미를 가지려면 잔차에 대역 밖 성분이 실제로 남아있어야 하므로,
    이 함수는 --no-bpf 전역 옵션과 무관하게 항상 BPF를 건너뛴다.
    """
    y = np.asarray(x, dtype=float).copy()
    if do_detrend and len(y) >= 3:
        y = detrend_tarvainen(y, Lambda)
    return znorm(y)


def chunkify(sig, seg_len):
    """1D 신호를 non-overlapping seg_len 청크로 자른다. 나머지는 버린다."""
    n = (len(sig) // seg_len) * seg_len
    if n == 0:
        return np.empty((0, seg_len))
    return np.asarray(sig[:n], dtype=float).reshape(-1, seg_len)


# ══════════════════════════════════════════════════════════════════
# 데이터 로딩 — fold별 예측을 participant 단위로 재조립 (R1-7)
# ══════════════════════════════════════════════════════════════════

def discover_models(runs_dir):
    models = set()
    for p in Path(runs_dir).glob("*_f*_info.json"):
        stem = p.name[: -len("_info.json")]
        model = stem.rsplit("_f", 1)[0]
        models.add(model)
    return sorted(models)


def load_participant_data(runs_dir, label_full, model, fs, seg_len,
                          do_detrend, do_bpf, Lambda):
    """
    반환: {participant_id: {"label": (n_chunk, seg_len),        # BWMD/SRE/WCR용(BPF)
                            "pred": (n_chunk, seg_len),
                            "label_edd": (n_chunk, seg_len),     # EDD 전용(non-BPF)
                            "pred_edd": (n_chunk, seg_len),
                            "n_rec": int, "folds": set}}
    K-fold 성질상 participant는 정확히 하나의 fold에서만 test로 등장한다.
    """
    runs_dir = Path(runs_dir)
    per_subj_label = defaultdict(list)
    per_subj_pred = defaultdict(list)
    per_subj_label_edd = defaultdict(list)
    per_subj_pred_edd = defaultdict(list)
    per_subj_nrec = defaultdict(int)
    per_subj_folds = defaultdict(set)

    info_files = sorted(runs_dir.glob(f"{model}_f*_info.json"))
    if not info_files:
        return {}

    for inf_p in info_files:
        info = json.load(open(inf_p))
        fold = info.get("fold")
        pred_p = runs_dir / f"{model}_f{fold}_test_pred.npy"
        if not pred_p.exists():
            raise FileNotFoundError(f"예측 파일 없음: {pred_p}")
        preds = np.load(pred_p, allow_pickle=True)

        ranges = info["test_ranges"]
        subj_of = info["test_subject_of_range"]
        if not (len(preds) == len(ranges) == len(subj_of)):
            raise ValueError(
                f"{model} fold{fold}: 개수 불일치 "
                f"(pred {len(preds)} / ranges {len(ranges)} / subjects {len(subj_of)})")

        for rec_i, ((s, e), subj) in enumerate(zip(ranges, subj_of)):
            p_rec = np.asarray(preds[rec_i], dtype=float).ravel()
            l_rec = np.asarray(label_full[s:e], dtype=float).ravel()
            L = min(len(p_rec), len(l_rec))
            if L < seg_len:
                continue
            p_ch = chunkify(p_rec[:L], seg_len)
            l_ch = chunkify(l_rec[:L], seg_len)
            n = min(len(p_ch), len(l_ch))
            if n == 0:
                continue
            for c in range(n):
                per_subj_pred[subj].append(
                    preprocess_chunk(p_ch[c], fs, do_detrend, do_bpf, Lambda))
                per_subj_label[subj].append(
                    preprocess_chunk(l_ch[c], fs, do_detrend, do_bpf, Lambda))
                per_subj_pred_edd[subj].append(
                    preprocess_chunk_edd(p_ch[c], fs, do_detrend, Lambda))
                per_subj_label_edd[subj].append(
                    preprocess_chunk_edd(l_ch[c], fs, do_detrend, Lambda))
            per_subj_nrec[subj] += 1
            per_subj_folds[subj].add(fold)

    out = {}
    for subj in sorted(per_subj_label):
        out[subj] = {
            "label": np.array(per_subj_label[subj]),
            "pred": np.array(per_subj_pred[subj]),
            "label_edd": np.array(per_subj_label_edd[subj]),
            "pred_edd": np.array(per_subj_pred_edd[subj]),
            "n_rec": per_subj_nrec[subj],
            "folds": sorted(per_subj_folds[subj]),
        }
    return out


# ══════════════════════════════════════════════════════════════════
# 컴포넌트 1: BWMD — 형태학 (DTW + peak timing) / amplitude overlap
# ══════════════════════════════════════════════════════════════════

def dtw_distance(a, b, sakoe_chiba_ratio=0.1):
    """
    Sakoe-Chiba band 제약 DTW. 무차원화(R1-3):
      (i) 진폭: 각 신호를 max|·|로 나눠 [-1,1]로 스케일 제거
      (ii) 경로길이: 누적비용을 (na+nb)로 나눠 길이 의존 제거
    두 정규화의 결과로 dtw_norm ∈ [0, 2]로 유계.
    """
    na, nb = len(a), len(b)
    band = max(1, int(sakoe_chiba_ratio * max(na, nb)))

    if USE_FASTDTW:
        dist, _ = _fastdtw(a, b, radius=band, dist=lambda x, y: abs(x - y))
        return float(dist) / max(na + nb, 1)

    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        j_lo, j_hi = max(1, i - band), min(nb, i + band)
        for j in range(j_lo, j_hi + 1):
            cost = abs(a[i - 1] - b[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[na, nb]) / max(na + nb, 1)


def peak_times(sig, fs):
    """수축기 피크 시각(초). 최소 간격 0.4s = 150bpm 상한."""
    peaks, _ = find_peaks(sig, distance=max(1, int(0.4 * fs)))
    return peaks / float(fs)


def median_rr(pt):
    return float(np.median(np.diff(pt))) if len(pt) >= 2 else None


def compute_bwmd(y, yhat, fs):
    """
    BWMD = (DTW_norm + λp·PeakDiff_norm) / (1 + λa·AreaOverlap)

    PeakDiff_norm은 RR 1주기로 나눠 무차원화하고 [0,1]로 클램프한다(R1-3).
    1.0 = "피크가 심박 한 주기만큼 어긋남" = 위상정보 완전 상실.
    AreaOverlap은 IoU(0~1)라 분모 ∈ [1, 1+λa].
    """
    a = y / (np.max(np.abs(y)) + 1e-8)
    b = yhat / (np.max(np.abs(yhat)) + 1e-8)
    dtw_norm = dtw_distance(a, b)

    pt_y, pt_h = peak_times(y, fs), peak_times(yhat, fs)
    if len(pt_y) >= 2 and len(pt_h) >= 2:
        k = min(len(pt_y), len(pt_h))
        diff_mean = float(np.mean(np.abs(pt_y[:k] - pt_h[:k])))
        rr = median_rr(pt_y)
        rr_norm = rr if (rr and rr > 1e-6) else (len(y) / float(fs))
        peak_norm = diff_mean / rr_norm
    elif len(pt_y) >= 1 and len(pt_h) >= 1:
        peak_norm = abs(pt_y[0] - pt_h[0]) / (len(y) / float(fs))
    else:
        peak_norm = 0.5      # 피크 검출 실패: 중립값
    peak_norm = float(np.clip(peak_norm, 0.0, 1.0))

    abs_y, abs_h = np.abs(y), np.abs(yhat)
    inter = float(_trapz(np.minimum(abs_y, abs_h)))
    union = float(_trapz(np.maximum(abs_y, abs_h)))
    area_overlap = inter / (union + 1e-8)

    bwmd = (dtw_norm + BWMD_LAMBDA_P * peak_norm) / (1.0 + BWMD_LAMBDA_A * area_overlap)
    return float(bwmd), float(dtw_norm), peak_norm, float(area_overlap)


# ══════════════════════════════════════════════════════════════════
# 컴포넌트 2: SRE — SNR 가중 오차 (부호 수정판, X-4)
# ══════════════════════════════════════════════════════════════════

def spectral_snr(y, fs, band=CARDIAC_BAND):
    """참조신호의 대역내/대역외 파워비(dB). prediction과 무관 = 캡처 난이도 지표."""
    nperseg = min(256, len(y))
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(y, fs=fs, nperseg=nperseg)
    m = (freqs >= band[0]) & (freqs <= band[1])
    sig_p = _trapz(psd[m], freqs[m])
    noi_p = _trapz(psd[~m], freqs[~m])
    if noi_p < 1e-12:
        return 100.0
    if sig_p < 1e-12:
        return -100.0
    return float(10.0 * np.log10(sig_p / noi_p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def snr_weight_g(snr_db):
    """
    [X-4 수정] 고SNR 항이 뺄셈이다.

      g = 1 + w_low·σ(SNR_low − s)·r_low − w_high·σ(s − SNR_high)·r_high
      r_low  = max(0, (SNR_low − s)/SNR_low)
      r_high = max(0, (s − SNR_high)/SNR_high)
      g ← clip(g, G_MIN, G_MAX)

    해석: 참조 PPG의 대역내 SNR이 낮으면(측정 자체가 어려운 구간) 동일 RMSE라도
    더 무겁게 보고, SNR이 충분히 높은 쉬운 구간에서는 가중을 낮춘다.
    (기존 코드는 두 항이 모두 덧셈이라 고SNR일수록 페널티가 커졌다.)

    검증값: g(3dB이하 시작)=1.00, g(20dB)=0.83, g(25dB)=0.67,
            g(30dB)=0.50, g(40dB)=0.20(하한), g(-10dB)=3.00(상한)
    """
    r_low = max(0.0, (SRE_SNR_LOW - snr_db) / SRE_SNR_LOW)
    r_high = max(0.0, (snr_db - SRE_SNR_HIGH) / SRE_SNR_HIGH)
    cri = sigmoid(SRE_SNR_LOW - snr_db)     # critical (저SNR) 게이트
    si = sigmoid(snr_db - SRE_SNR_HIGH)     # saturation (고SNR) 게이트
    g = 1.0 + SRE_W_LOW * cri * r_low - SRE_W_HIGH * si * r_high
    return float(np.clip(g, SRE_G_MIN, SRE_G_MAX)), float(cri), float(si)


def compute_sre(y, yhat, fs):
    """SRE = RMSE × g(SNR_ref).  z-norm 경로에서 RMSE ∈ [0,2]."""
    e = np.ravel(yhat) - np.ravel(y)
    rmse = float(np.sqrt(np.mean(e ** 2))) if e.size else 0.0
    snr_db = spectral_snr(y, fs)
    g, cri, si = snr_weight_g(snr_db)
    return float(rmse * g), rmse, snr_db, g, cri, si


# ══════════════════════════════════════════════════════════════════
# 컴포넌트 3: WCR — 일치도 (CCC + Spearman)
# ══════════════════════════════════════════════════════════════════

def concordance_ccc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    mx, my = np.mean(x), np.mean(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    cov = np.cov(x, y, ddof=1)[0, 1]
    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0:
        return 0.0
    return float(np.clip(2.0 * cov / denom, -1.0, 1.0))


def compute_wcr(y, yhat, beta=WCR_BETA):
    x, z = np.ravel(y), np.ravel(yhat)
    ccc = concordance_ccc(x, z)
    rho, _ = spearmanr(x, z)
    if np.isnan(rho):
        rho = 0.0
    return float(np.clip(beta * ccc + (1 - beta) * rho, -1.0, 1.0)), ccc, float(rho)


# ══════════════════════════════════════════════════════════════════
# 컴포넌트 4: EDD — 잔차 분포의 정규성 이탈 (JSD)
# ══════════════════════════════════════════════════════════════════

def compute_edd(y_nobpf, yhat_nobpf, fs, band=CARDIAC_BAND):
    """
    EDD v2 — 잔차의 심장대역 파워 비율 (Error-band Density / Deviation).

    e = yhat − y (BPF를 거치지 않은, detrend+z-norm만 된 신호에서 계산)
    EDD = (e의 PSD 중 심장대역 파워) / (e의 전체 파워)

    해석: 낮을수록 좋음(lower-is-better). 잔차 에너지가 심장대역 **밖**에
    퍼져있다는 뜻 = 모델이 놓친 부분이 비구조적 잡음(센서노이즈, 타이밍 지터 등)
    이라는 뜻. 반대로 잔차가 심장대역 **안**에 몰려있으면, 예측이 진짜와는
    다른(위상이 틀렸거나 심박수 자체가 틀린) '유령 맥파' 구조를 갖고 있다는
    뜻이라 나쁘다.

    정의상 파워비율이라 [0,1]로 자연히 유계 — anchor는 이론경계 그대로 쓴다.

    ※ v1(잔차 히스토그램의 정규성 이탈, JSD)은 실측(UBFC) 결과 POS가 전 모델
    중 EDD 최고점을 받는 역전 현상이 나왔다. 잔차의 '가우시안성'은 충실도와
    무관하다는 게 확인됐다(엉터리 예측이라도 두 신호 차는 중심극한정리로
    오히려 가우시안에 가까워짐). v2는 이 문제를 피하려고 분포 형상 대신
    '어디에 에너지가 남아있는가'를 직접 본다.

    ★ 이 함수에 넣는 y_nobpf/yhat_nobpf는 반드시 preprocess_chunk_edd()를
    거친(BPF 미적용) 신호여야 한다. BWMD/SRE/WCR용 BPF 신호를 넣으면 분모까지
    이미 대역 안에 갇혀있어 비율이 항상 ~1.0으로 나와 무의미해진다.
    """
    e = np.ravel(yhat_nobpf) - np.ravel(y_nobpf)
    nperseg = min(256, e.size)
    if nperseg < 16:
        return 0.0
    freqs, psd = welch(e, fs=fs, nperseg=nperseg)
    m = (freqs >= band[0]) & (freqs <= band[1])
    band_power = _trapz(psd[m], freqs[m])
    total_power = _trapz(psd, freqs)
    if total_power < 1e-12:
        return 0.0                       # 잔차 ≈ 0 → 이탈 없음(퇴화 케이스)
    return float(np.clip(band_power / total_power, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════
# 보조 지표: HR (R1-6), PRV (R1-5, 용어·단위 교정)
# ══════════════════════════════════════════════════════════════════

def hr_from_psd(sig, fs, band=CARDIAC_BAND):
    """PSD 최대 피크 주파수로부터 HR(bpm). 대역 내에서만 탐색."""
    nperseg = min(256, len(sig))
    if nperseg < 16:
        return np.nan
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(m):
        return np.nan
    return float(freqs[m][np.argmax(psd[m])] * 60.0)


def prv_from_chunks(chunks, fs):
    """
    PRV(pulse rate variability) — 용어 교정: 접촉 ECG의 HRV가 아니라
    맥파 기반이므로 PRV가 정확하다(R1-5).
    단위: ms (pNN50만 %).

    ★ RR 간격은 각 chunk **내부**에서만 계산한다. chunk를 이어붙인 뒤 피크를 잡으면
      경계에서 실제로는 존재하지 않는 RR 간격이 생겨 SDNN/RMSSD를 오염시킨다.
    ★ 10초 chunk는 RR 표본이 대략 8~15개뿐이라 SDNN/RMSSD의 신뢰구간이 넓다.
      본 지표는 컴포넌트 근거(diagnostic)로만 보고하고 RPFI 점수에는 들어가지 않는다.
    """
    rr_all, n_short = [], 0
    for ch in chunks:
        pt = peak_times(ch, fs)
        if len(pt) < 2:
            n_short += 1
            continue
        rr_all.append(np.diff(pt))
    if not rr_all:
        return {k: np.nan for k in
                ["mean_rr_ms", "sdnn_ms", "rmssd_ms", "pnn50_pct", "mean_hr_bpm", "n_rr"]}
    rr = np.concatenate(rr_all)
    drr = np.concatenate([np.diff(r) for r in rr_all if len(r) > 1]) \
        if any(len(r) > 1 for r in rr_all) else np.array([])
    mean_rr = float(np.mean(rr))
    return {
        "mean_rr_ms": mean_rr * 1000.0,
        "sdnn_ms": float(np.std(rr, ddof=1)) * 1000.0 if rr.size > 1 else 0.0,
        "rmssd_ms": float(np.sqrt(np.mean(drr ** 2))) * 1000.0 if drr.size else 0.0,
        "pnn50_pct": 100.0 * float(np.mean(np.abs(drr) > 0.05)) if drr.size else 0.0,
        "mean_hr_bpm": 60.0 / mean_rr if mean_rr > 0 else np.nan,
        "n_rr": int(rr.size),
    }


def compute_psnr(ref, pred):
    """
    PSNR(dB). MAX 정의 명시(R1-M1): z-normalize된 참조신호의 최대 절댓값을 MAX로 쓴다
    (영상용 255 같은 고정 동적범위가 없으므로).
    """
    mse = float(np.mean((ref - pred) ** 2))
    if mse == 0:
        return float("inf")
    max_val = float(np.max(np.abs(ref)))
    return float(10.0 * np.log10((max_val ** 2) / mse))


# ══════════════════════════════════════════════════════════════════
# 정규화 & 집계 (고정 anchor + 비보상적, R1-2)
# ══════════════════════════════════════════════════════════════════

def normalize_component(name, value, anchors):
    """컴포넌트 원값 → [0,1] 품질점수(1=최고). 풀 비의존."""
    lo, hi = anchors[name]
    if hi <= lo:
        return 0.0
    t = float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
    return 1.0 - t if DIRECTION[name] == "lower" else t


def aggregate(scores, weights, mode="geometric", eps=1e-6):
    """
    scores: {comp: [0,1] 품질점수}
    mode:
      "arithmetic" — 가중합. 보상적(한 축이 0이어도 나머지로 상쇄).
      "geometric"  — 가중 기하평균. 비보상적: 한 축이 0에 가까우면 전체가 0으로 끌린다.
    """
    w_sum = sum(weights[c] for c in COMPONENTS)
    if mode == "arithmetic":
        agg = sum(weights[c] * scores[c] for c in COMPONENTS) / w_sum
    else:
        log_sum = sum(weights[c] * np.log(max(scores[c], eps)) for c in COMPONENTS)
        agg = float(np.exp(log_sum / w_sum))
    return float(np.clip(agg, 0.0, 1.0) * 100.0)


# ══════════════════════════════════════════════════════════════════
# 통계 — participant 단위 cluster bootstrap, 사후검정 (R1-7)
# ══════════════════════════════════════════════════════════════════

def cluster_bootstrap_ci(values, n_boot=2000, ci=95, seed=42):
    """participant를 클러스터로 리샘플링. values는 participant당 1개 값."""
    vals = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if vals.size < 2:
        m = float(vals.mean()) if vals.size else np.nan
        return m, m, m
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boots = vals[idx].mean(axis=1)
    a = (100 - ci) / 2
    return (float(vals.mean()),
            float(np.percentile(boots, a)),
            float(np.percentile(boots, 100 - a)))


def holm_adjust(pvals):
    """Holm-Bonferroni 보정."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def posthoc_wilcoxon(model_scores, models):
    """
    Friedman 유의 시 pairwise Wilcoxon signed-rank + Holm 보정.
    model_scores: {model: participant별 RPFI 리스트} (participant 정렬 동일 가정)
    """
    pairs, stats, praw = [], [], []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a = np.asarray(model_scores[models[i]], dtype=float)
            b = np.asarray(model_scores[models[j]], dtype=float)
            m = ~(np.isnan(a) | np.isnan(b))
            if m.sum() < 6 or np.allclose(a[m], b[m]):
                continue
            try:
                s, p = wilcoxon(a[m], b[m])
            except ValueError:
                continue
            pairs.append((models[i], models[j]))
            stats.append(float(s))
            praw.append(float(p))
    if not pairs:
        return []
    padj = holm_adjust(praw)
    return [(pa[0], pa[1], st, pr, pj, float(np.nanmean(model_scores[pa[0]]) -
                                             np.nanmean(model_scores[pa[1]])))
            for pa, st, pr, pj in zip(pairs, stats, praw, padj)]


# ══════════════════════════════════════════════════════════════════
# 민감도 / 중복성 (R1-11, R2)
# ══════════════════════════════════════════════════════════════════

def weight_sensitivity(per_model_comp_means, anchors, base_weights,
                       mode, n_draw=500, seed=42):
    """
    Dirichlet로 가중치를 무작위 추출해 모델 순위가 얼마나 흔들리는지 본다.
    반환: (기준 순위와의 Spearman ρ 분포, top-1 유지율)
    """
    rng = np.random.default_rng(seed)
    models = list(per_model_comp_means)

    def rank_with(w):
        sc = []
        for m in models:
            s = {c: normalize_component(c, per_model_comp_means[m][c], anchors)
                 for c in COMPONENTS}
            sc.append(aggregate(s, w, mode))
        return np.asarray(sc)

    base = rank_with(base_weights)
    base_top = models[int(np.argmax(base))]
    alpha = np.array([base_weights[c] for c in COMPONENTS]) * 20.0
    rhos, top_same = [], 0
    for _ in range(n_draw):
        draw = rng.dirichlet(alpha)
        w = {c: float(draw[i]) for i, c in enumerate(COMPONENTS)}
        cur = rank_with(w)
        r, _ = spearmanr(base, cur)
        if not np.isnan(r):
            rhos.append(r)
        if models[int(np.argmax(cur))] == base_top:
            top_same += 1
    return np.asarray(rhos), top_same / float(n_draw), base_top


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True, help="runs/{dataset}_cv{K} 디렉토리")
    ap.add_argument("--label", required=True, help="데이터셋 전체 label npy")
    ap.add_argument("--models", default=None, help="콤마 구분. 미지정시 자동 탐색")
    ap.add_argument("--out", default="results_eval")
    ap.add_argument("--fs", type=int, default=None, help="미지정시 데이터셋 기본값")
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--agg", choices=["geometric", "arithmetic"], default="geometric",
                    help="geometric=비보상적(기본), arithmetic=기존 가중합")
    ap.add_argument("--anchors", default=None, help="anchor JSON 파일로 덮어쓰기")
    ap.add_argument("--weights", default=None, help="가중치 JSON 파일로 덮어쓰기")
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true", help="대역통과 민감도 확인용")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-multi-fold", action="store_true",
                    help="participant가 여러 fold에 걸쳐 나타나도 경고만 하고 진행. "
                        "cross_dataset_eval.py 산출물을 정상적으로 채점할 때만 켤 것. "
                        "within-dataset 결과인데 이게 필요하다면 --runs 경로가 잘못됐을 가능성이 큼.")
    args = ap.parse_args()

    fs = args.fs or FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    do_detrend = not args.no_detrend
    do_bpf = not args.no_bpf

    anchors = dict(DEFAULT_ANCHORS)
    if args.anchors:
        anchors.update({k: tuple(v) for k, v in json.load(open(args.anchors)).items()})
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.load(open(args.weights)))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_full = np.load(args.label)
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else discover_models(args.runs))
    if not models:
        raise SystemExit(f"{args.runs} 에서 모델을 찾지 못했습니다.")

    print("=" * 78)
    print(f"RPFI 평가 — dataset={args.dataset}  fs={fs}Hz  "
          f"chunk={seg_len} samples ({args.eval_seconds}s)")
    print(f"  전처리: detrend={do_detrend}(λ={args.lambda_detrend})  "
          f"bandpass={do_bpf}{CARDIAC_BAND if do_bpf else ''}  z-norm=True (label/pred 동일)")
    print(f"  집계: {args.agg}{'  (비보상적)' if args.agg == 'geometric' else '  (보상적/기존)'}"
          f"   가중치: {weights}")
    print(f"  anchor(고정): " + ", ".join(f"{c}{anchors[c]}" for c in COMPONENTS))
    print(f"  DTW backend: {'fastdtw' if USE_FASTDTW else 'naive+Sakoe-Chiba'}")
    print(f"  모델 {len(models)}개: {', '.join(models)}")
    print("=" * 78)

    timing = defaultdict(list)
    rows = []                       # participant × model 전면 기록
    per_model_participant = defaultdict(dict)   # model -> {subj: rpfi}
    per_model_comp_vals = defaultdict(lambda: defaultdict(list))
    raw_component_pool = defaultdict(list)      # anchor 검증용

    for model in models:
        data = load_participant_data(args.runs, label_full, model, fs, seg_len,
                                     do_detrend, do_bpf, args.lambda_detrend)
        if not data:
            print(f"  [건너뜀] {model}: 산출물 없음")
            continue

        # ── 안전장치: within-dataset K-fold라면 participant는 정확히 1개 fold에서만
        # test로 등장해야 한다. 2개 이상이면 cross_dataset_eval.py 산출물(source의
        # 5개 fold 체크포인트를 target 전체에 반복 적용)을 잘못된 --runs 경로로 채점
        # 중일 가능성이 매우 크다(실제로 2026-07-27 이런 사고가 있었음). ──
        multi = {s: d["folds"] for s, d in data.items() if len(d["folds"]) > 1}
        if multi and not args.allow_multi_fold:
            sample = list(multi.items())[:3]
            raise SystemExit(
                f"\n⚠️  {model}: participant {len(multi)}명이 fold 여러 개에 걸쳐 나타납니다 "
                f"(예: {sample}).\n"
                f"    within-dataset K-fold라면 한 participant는 정확히 1개 fold에서만 "
                f"test여야 합니다.\n"
                f"    --runs='{args.runs}' 가 실제로는 cross_dataset_eval.py 산출물"
                f"(예: runs/cross_X_to_{args.dataset})일 가능성이 큽니다 — 경로를 확인하세요.\n"
                f"    cross-dataset 결과를 의도적으로 채점하는 거라면 --allow-multi-fold를 붙이세요.")
        elif multi:
            print(f"  ℹ️  {model}: participant {len(multi)}명이 여러 fold에 걸쳐 나타남 "
                  f"(--allow-multi-fold로 허용됨 — cross-dataset 채점이 맞는지 확인할 것)")

        for subj, d in sorted(data.items()):
            L, P = d["label"], d["pred"]
            L_e, P_e = d["label_edd"], d["pred_edd"]
            n_chunk = len(L)
            if n_chunk == 0:
                continue

            # ── chunk 단위: BWMD, HR ──
            bwmds, hr_l, hr_p = [], [], []
            t0 = time.perf_counter()
            for c in range(n_chunk):
                b, _, _, _ = compute_bwmd(L[c], P[c], fs)
                bwmds.append(b)
            timing["BWMD"].append((time.perf_counter() - t0) / n_chunk)
            for c in range(n_chunk):
                hr_l.append(hr_from_psd(L[c], fs))
                hr_p.append(hr_from_psd(P[c], fs))

            y_cat, p_cat = L.ravel(), P.ravel()
            y_cat_e, p_cat_e = L_e.ravel(), P_e.ravel()

            t0 = time.perf_counter(); wcr, ccc, rho = compute_wcr(y_cat, p_cat)
            timing["WCR"].append(time.perf_counter() - t0)
            t0 = time.perf_counter(); edd = compute_edd(y_cat_e, p_cat_e, fs)
            timing["EDD"].append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            sre, rmse, snr_db, g, cri, si = compute_sre(y_cat, p_cat, fs)
            timing["SRE"].append(time.perf_counter() - t0)

            bwmd = float(np.mean(bwmds))
            comp = {"BWMD": bwmd, "SRE": sre, "WCR": wcr, "EDD": edd}
            scores = {c: normalize_component(c, comp[c], anchors) for c in COMPONENTS}
            rpfi = aggregate(scores, weights, args.agg)
            rpfi_alt = aggregate(scores, weights,
                                 "arithmetic" if args.agg == "geometric" else "geometric")

            hr_l_a = np.asarray(hr_l, dtype=float)
            hr_p_a = np.asarray(hr_p, dtype=float)
            hm = ~(np.isnan(hr_l_a) | np.isnan(hr_p_a))
            hr_mae = float(np.mean(np.abs(hr_l_a[hm] - hr_p_a[hm]))) if hm.any() else np.nan

            prv_l = prv_from_chunks(L, fs)
            prv_p = prv_from_chunks(P, fs)

            for c in COMPONENTS:
                per_model_comp_vals[model][c].append(comp[c])
                raw_component_pool[c].append(comp[c])
            per_model_participant[model][subj] = rpfi

            rows.append(dict(
                dataset=args.dataset, model=model, participant=subj,
                n_recordings=d["n_rec"], n_chunks=n_chunk, folds="|".join(map(str, d["folds"])),
                BWMD=bwmd, SRE=sre, WCR=wcr, EDD=edd,
                s_BWMD=scores["BWMD"], s_SRE=scores["SRE"],
                s_WCR=scores["WCR"], s_EDD=scores["EDD"],
                RPFI=rpfi, RPFI_alt_agg=rpfi_alt,
                RMSE=rmse, PSNR_dB=compute_psnr(y_cat, p_cat),
                SNR_ref_dB=snr_db, g_SNR=g, cri=cri, si=si,
                CCC=ccc, Spearman=rho,
                HR_label_bpm=float(np.nanmean(hr_l_a)), HR_pred_bpm=float(np.nanmean(hr_p_a)),
                HR_MAE_bpm=hr_mae,
                PRV_sdnn_label_ms=prv_l["sdnn_ms"], PRV_sdnn_pred_ms=prv_p["sdnn_ms"],
                PRV_rmssd_label_ms=prv_l["rmssd_ms"], PRV_rmssd_pred_ms=prv_p["rmssd_ms"],
                PRV_n_rr_label=prv_l["n_rr"], PRV_n_rr_pred=prv_p["n_rr"],
            ))

        print(f"  [완료] {model:16s} participants={len(data)}  "
              f"chunks={sum(len(d['label']) for d in data.values())}")

    if not rows:
        raise SystemExit("집계된 결과가 없습니다.")

    # ── anchor 타당성 점검 ──
    print("\n" + "=" * 78)
    print("Anchor 점검 (실측 범위가 고정 anchor를 벗어나면 클리핑 = 정보 손실)")
    print("=" * 78)
    print(f"  {'Comp':>6s} | {'anchor_lo':>10s} {'anchor_hi':>10s} | "
          f"{'obs_p1':>10s} {'obs_p99':>10s} | {'clipped':>10s}")
    any_clip = False
    for c in COMPONENTS:
        v = np.asarray(raw_component_pool[c])
        lo, hi = anchors[c]
        n_clip = int(((v < lo) | (v > hi)).sum())
        any_clip |= n_clip > 0
        flag = "" if n_clip == 0 else "  ⚠"
        print(f"  {c:>6s} | {lo:>10.4f} {hi:>10.4f} | "
              f"{np.percentile(v, 1):>10.4f} {np.percentile(v, 99):>10.4f} | "
              f"{n_clip:>5d}/{v.size}{flag}")
    if any_clip:
        print("\n  ⚠ 클리핑 발생. anchor는 '평가 대상 모델 풀'이 아니라 외부 기준으로")
        print("    정해야 하므로(R1-2), 이 실행 결과로 anchor를 다시 맞추면 안 됩니다.")
        print("    SRE anchor는 g(SNR) 부호를 수정하면서 척도가 바뀌었으므로,")
        print("    rpfi_degradation.py를 수정된 compute_sre로 재실행해 p1-p99를")
        print("    다시 뽑은 뒤 --anchors 로 주입하십시오.")

    # ── 모델 요약 (participant cluster bootstrap CI) ──
    print("\n" + "=" * 78)
    print(f"모델별 요약 — participant 단위 cluster bootstrap 95% CI (n_boot={args.n_boot})")
    print("=" * 78)

    summary_rows, ranking = [], {}
    for model in models:
        if model not in per_model_participant:
            continue
        subs = sorted(per_model_participant[model])
        rpfis = [per_model_participant[model][s] for s in subs]
        r_m, r_lo, r_hi = cluster_bootstrap_ci(rpfis, args.n_boot, seed=args.seed)
        ranking[model] = r_m

        line = {"model": model, "n_participants": len(subs),
                "RPFI": r_m, "RPFI_lo": r_lo, "RPFI_hi": r_hi}
        print(f"\n>> {model}   (participants={len(subs)})")
        print(f"   RPFI : {r_m:6.2f}   95%CI [{r_lo:6.2f}, {r_hi:6.2f}]")
        for c in COMPONENTS:
            m_, lo_, hi_ = cluster_bootstrap_ci(per_model_comp_vals[model][c],
                                                args.n_boot, seed=args.seed)
            arrow = "↓" if DIRECTION[c] == "lower" else "↑"
            print(f"   {c:5s}: {m_:8.5f}  95%CI [{lo_:8.5f}, {hi_:8.5f}]  ({arrow})")
            line[c] = m_; line[f"{c}_lo"] = lo_; line[f"{c}_hi"] = hi_

        mrows = [r for r in rows if r["model"] == model]
        for key, lbl in [("HR_MAE_bpm", "HR MAE (bpm)"), ("RMSE", "RMSE"),
                         ("PSNR_dB", "PSNR (dB)"), ("SNR_ref_dB", "SNR_ref (dB)")]:
            v = np.asarray([r[key] for r in mrows], dtype=float)
            line[key] = float(np.nanmean(v))
            print(f"   {lbl:16s}: {np.nanmean(v):8.4f}")
        summary_rows.append(line)

    # ── 계산비용 (R2-6) ──
    print("\n" + "=" * 78)
    print("계산비용 (컴포넌트당, participant 1인 기준 / BWMD는 chunk 1개 기준)")
    print("=" * 78)
    print(f"  {'Component':>12s} | {'Mean(ms)':>10s} | {'Std(ms)':>10s} | {'Max(ms)':>10s}")
    for c in ["BWMD", "SRE", "WCR", "EDD"]:
        if timing[c]:
            a = np.asarray(timing[c]) * 1000
            print(f"  {c:>12s} | {a.mean():>10.3f} | {a.std():>10.3f} | {a.max():>10.3f}")

    # ── 컴포넌트 중복성 (R1-11, R2-7) ──
    print("\n" + "=" * 78)
    print("컴포넌트 상호상관 (Spearman) — 중복성 점검")
    print("=" * 78)
    pool = [np.asarray(raw_component_pool[c]) for c in COMPONENTS]
    print(f"{'':>8s}" + "".join(f"{c:>9s}" for c in COMPONENTS))
    for i, ci_ in enumerate(COMPONENTS):
        cells = []
        for j in range(len(COMPONENTS)):
            if i == j:
                cells.append(f"{1.0:>9.3f}")
            else:
                r, _ = spearmanr(pool[i], pool[j])
                cells.append(f"{0.0 if np.isnan(r) else r:>9.3f}")
        print(f"{ci_:>8s}" + "".join(cells))
    print("\n  |ρ|<0.3 독립 | 0.3-0.6 부분공유 | >0.6 중복 가능성")

    # ── 가중치 민감도 (R1-11) ──
    comp_means = {m: {c: float(np.mean(per_model_comp_vals[m][c])) for c in COMPONENTS}
                  for m in per_model_participant}
    if len(comp_means) >= 3:
        rhos, top_rate, base_top = weight_sensitivity(
            comp_means, anchors, weights, args.agg, seed=args.seed)
        print("\n" + "=" * 78)
        print("가중치 민감도 (Dirichlet 500회 재추출)")
        print("=" * 78)
        print(f"  기준 가중치 1위 모델: {base_top}")
        print(f"  순위 Spearman ρ: mean={rhos.mean():.4f}  min={rhos.min():.4f}  "
              f"p5={np.percentile(rhos, 5):.4f}")
        print(f"  1위 모델 유지율: {100 * top_rate:.1f}%")
        print("  → ρ가 1에 가깝고 유지율이 높을수록 가중치 선택에 강건한 랭킹입니다.")

    # ── Friedman + 사후검정 (R1-7) ──
    print("\n" + "=" * 78)
    print("모델 간 차이의 통계적 유의성 (분석 단위 = participant)")
    print("=" * 78)
    valid = [m for m in models if m in per_model_participant]
    common = sorted(set.intersection(*[set(per_model_participant[m]) for m in valid])) \
        if valid else []
    print(f"  모델 {len(valid)}개, 공통 participant N={len(common)}")
    print(f"  Friedman χ² 이론 상한 = N×(k−1) = {len(common)}×{len(valid) - 1} "
          f"= {len(common) * (len(valid) - 1)}"
          if len(valid) >= 2 else "")

    posthoc = []
    if len(valid) >= 3 and len(common) >= 5:
        groups = [[per_model_participant[m][s] for s in common] for m in valid]
        stat, p = friedmanchisquare(*groups)
        print(f"  Friedman χ² = {stat:.4f},  p = {p:.6g}")
        if p < 0.05:
            print("  → 유의 (p<0.05). 사후검정 진행.\n")
            posthoc = posthoc_wilcoxon(
                {m: [per_model_participant[m][s] for s in common] for m in valid}, valid)
            print(f"  {'model A':>16s} {'model B':>16s} {'ΔRPFI':>9s} "
                  f"{'p_raw':>10s} {'p_Holm':>10s}  sig")
            for a, b, _s, pr, pj, dlt in sorted(posthoc, key=lambda x: x[4]):
                print(f"  {a:>16s} {b:>16s} {dlt:>+9.2f} {pr:>10.4g} {pj:>10.4g}"
                      f"  {'*' if pj < 0.05 else ''}")
            n_sig = sum(1 for x in posthoc if x[4] < 0.05)
            print(f"\n  Holm 보정 후 유의한 쌍: {n_sig}/{len(posthoc)}")
            if n_sig == 0:
                print("  ⚠ 전체 검정은 유의하나 개별 쌍은 유의하지 않습니다 — "
                      "모델 간 실질적 차이가 작다는 뜻입니다.")
        else:
            print("  → 유의하지 않음 (p≥0.05). 모델 간 RPFI 차이를 주장할 수 없습니다.")
    else:
        print("  모델 또는 participant 수 부족으로 검정 생략")

    # ── 최종 랭킹 ──
    print("\n" + "=" * 78)
    print(f"최종 RPFI 랭킹 (0-100, 높을수록 충실도 높음 / 집계={args.agg})")
    print("=" * 78)
    for i, (m, s) in enumerate(sorted(ranking.items(), key=lambda x: -x[1]), 1):
        row = next(r for r in summary_rows if r["model"] == m)
        print(f"  {i:2d}. {m:18s} RPFI={s:6.2f}  "
              f"[{row['RPFI_lo']:.2f}, {row['RPFI_hi']:.2f}]   "
              f"HR MAE={row['HR_MAE_bpm']:.2f} bpm")

    # ── CSV 저장 ──
    tag = f"{args.dataset}_{args.agg}" + ("" if do_bpf else "_nobpf")
    f1 = out_dir / f"rpfi_participant_{tag}.csv"
    with open(f1, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    f2 = out_dir / f"rpfi_summary_{tag}.csv"
    with open(f2, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)

    files = [f1, f2]
    if posthoc:
        f3 = out_dir / f"rpfi_posthoc_{tag}.csv"
        with open(f3, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["model_A", "model_B", "wilcoxon_stat", "p_raw", "p_holm", "delta_RPFI"])
            for a, b, s_, pr, pj, dl in posthoc:
                w.writerow([a, b, s_, pr, pj, dl])
        files.append(f3)

    cfg = dict(dataset=args.dataset, fs=fs, seg_len=seg_len,
               eval_seconds=args.eval_seconds, detrend=do_detrend,
               lambda_detrend=args.lambda_detrend, bandpass=do_bpf,
               cardiac_band=CARDIAC_BAND, aggregation=args.agg,
               weights=weights, anchors={k: list(v) for k, v in anchors.items()},
               sre_params=dict(snr_low=SRE_SNR_LOW, snr_high=SRE_SNR_HIGH,
                               w_low=SRE_W_LOW, w_high=SRE_W_HIGH,
                               g_min=SRE_G_MIN, g_max=SRE_G_MAX),
               models=list(ranking), n_boot=args.n_boot, seed=args.seed,
               dtw_backend="fastdtw" if USE_FASTDTW else "naive")
    f4 = out_dir / f"rpfi_config_{tag}.json"
    json.dump(cfg, open(f4, "w"), indent=2, default=str)
    files.append(f4)

    print("\n저장:")
    for f in files:
        print(f"  {f}")


if __name__ == "__main__":
    main()

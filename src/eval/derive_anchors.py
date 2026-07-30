"""
derive_anchors.py — RPFI 고정 anchor 재도출 + 알려진 열화에 대한 검증
=====================================================================
왜 다시 뽑아야 하나
--------------------
기존 anchor SRE[0.065, 6.493]는 g(SNR)의 saturation 항 부호가 뒤집혀 있던
상태(자체발견 X-4)에서 만든 열화 벤치마크 분포에서 나온 값이다.
rpfi_eval.py에서 부호를 고치면서 SRE의 척도 자체가 바뀌었으므로
(g(30dB): 1.50 → 0.50, g(40dB): 1.83 → 0.20) 옛 anchor는 무효다.

★ 이 스크립트는 rpfi_eval.py의 컴포넌트 함수를 **직접 import**한다.
  anchor를 만드는 코드와 쓰는 코드가 같은 함수를 공유하므로 척도 불일치가
  구조적으로 불가능하다. rpfi_eval.py를 고치면 반드시 이걸 다시 돌릴 것.

anchor 정의 (R1-2 대응)
------------------------
"평가 대상 모델 풀"이 아니라 외부 합성 기준으로 정한다. 두 개의 기준 앙상블:

  ideal — 예측이 참조와 사실상 동일 (잔차 std가 신호의 1%)
          → 도달 가능한 최선. RPFI 100의 조작적 정의.
  null  — 예측이 참조와 무관. 두 종류를 모두 포함:
          (a) 심박이 다른 독립 맥파   = "그럴듯하지만 틀린 생리신호"
          (b) 대역통과된 백색잡음      = "생리신호 아님"
          → RPFI 0의 조작적 정의.

이렇게 잡으면 anchor가 임의의 열화 스윕 설계에 의존하지 않고,
"100 = 참조와 구별 불가 / 0 = 무관한 신호와 다를 바 없음"으로 해석된다.

열화 검증 (R1-8 대응)
----------------------
13종 열화 × 6단계 × 반복으로 각 컴포넌트가 **의도한 열화에 단조 반응**하는지
확인하고, 반응 폭을 anchor 범위 대비 비율로 보고한다. 리뷰어가 지적한
"순환적 검증"(모델 성능으로 지표를 정당화)이 아니라, 정답을 아는 합성 열화에
대한 외부 검증이다.

사용법
------
    python derive_anchors.py --out anchors                 # 기본 (fs 30+20 통합)
    python derive_anchors.py --out anchors --reps 16       # 더 촘촘히
    python derive_anchors.py --out anchors --no-bpf        # rpfi_eval과 옵션 일치시킬 것

산출물
------
    anchors/rpfi_anchors.json          → rpfi_eval.py --anchors 로 주입
    anchors/anchor_ensembles.csv       → ideal/null 앙상블 원값
    anchors/degradation_sweep.csv      → 열화 스윕 전체 결과
    anchors/degradation_report.txt     → 단조성·민감도 요약
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import spearmanr, mannwhitneyu, ttest_ind

from rpfi_eval import (
    COMPONENTS, DIRECTION, SRE_G_MAX, CARDIAC_BAND,
    compute_bwmd, compute_sre, compute_wcr, compute_edd,
    preprocess_chunk, preprocess_chunk_edd,
)


# ══════════════════════════════════════════════════════════════════
# 합성 참조 PPG
# ══════════════════════════════════════════════════════════════════

def synth_ppg(n, fs, rng, hr_bpm=None):
    """
    생리학적으로 그럴듯한 맥파: 기본파 + 고조파(dicrotic notch) +
    호흡성 동성부정맥(RSA) + 완만한 진폭변조.
    """
    if hr_bpm is None:
        hr_bpm = rng.uniform(55, 100)
    t = np.arange(n) / fs
    f0 = hr_bpm / 60.0

    # RSA: 호흡(0.2~0.35Hz)에 따른 순간주파수 변조 → 위상 적분
    f_resp = rng.uniform(0.2, 0.35)
    inst_f = f0 * (1.0 + 0.06 * np.sin(2 * np.pi * f_resp * t + rng.uniform(0, 2 * np.pi)))
    phase = 2 * np.pi * np.cumsum(inst_f) / fs

    sig = (np.sin(phase)
           + 0.35 * np.sin(2 * phase + 0.8)      # dicrotic notch
           + 0.12 * np.sin(3 * phase + 1.6))
    sig *= (1.0 + 0.10 * np.sin(2 * np.pi * f_resp * t))   # 진폭변조
    return sig, hr_bpm


def _bp(x, fs, lo, hi, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [max(lo / nyq, 1e-6), min(hi / nyq, 0.99)], btype="bandpass")
    return filtfilt(b, a, x)


# ══════════════════════════════════════════════════════════════════
# 열화 13종 — level ∈ [0,1] (0=열화없음, 1=최대)
# ══════════════════════════════════════════════════════════════════

def deg_white_noise(y, fs, lv, rng):
    return y + (2.0 * lv) * np.std(y) * rng.standard_normal(len(y))


def deg_colored_noise(y, fs, lv, rng):
    w = rng.standard_normal(len(y))
    f = np.fft.rfftfreq(len(y), 1 / fs)
    spec = np.fft.rfft(w) / np.maximum(f, 1e-3)      # 1/f
    c = np.fft.irfft(spec, n=len(y))
    c = c / (np.std(c) + 1e-12)
    return y + (2.0 * lv) * np.std(y) * c


def deg_baseline_wander(y, fs, lv, rng):
    t = np.arange(len(y)) / fs
    fw = rng.uniform(0.05, 0.3)
    return y + (3.0 * lv) * np.std(y) * np.sin(2 * np.pi * fw * t + rng.uniform(0, 6.28))


def deg_amplitude_scale(y, fs, lv, rng):
    return y * (1.0 + 4.0 * lv)                       # z-norm이 지울 것으로 예상


def deg_dc_offset(y, fs, lv, rng):
    return y + (5.0 * lv) * np.std(y)                 # z-norm이 지울 것으로 예상


def _spikes(y, lv, rng, idx):
    z = y.copy()
    amp = 4.0 * np.std(y)
    z[idx] += amp * rng.choice([-1, 1], size=len(idx))
    return z


def deg_spike_burst(y, fs, lv, rng):
    k = max(1, int(0.05 * lv * len(y)))
    start = rng.integers(0, max(1, len(y) - k))
    return _spikes(y, lv, rng, np.arange(start, min(start + k, len(y))))


def deg_spike_dispersed(y, fs, lv, rng):
    k = max(1, int(0.05 * lv * len(y)))
    return _spikes(y, lv, rng, rng.choice(len(y), size=k, replace=False))


def deg_dropout(y, fs, lv, rng):
    z = y.copy()
    k = max(1, int(0.3 * lv * len(y)))
    start = rng.integers(0, max(1, len(y) - k))
    z[start:start + k] = 0.0
    return z


def deg_temporal_jitter(y, fs, lv, rng):
    n = len(y)
    t = np.arange(n)
    warp = t + (0.15 * lv * fs) * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t / fs)
    return np.interp(t, np.clip(warp, 0, n - 1), y)


def deg_phase_shift(y, fs, lv, rng):
    s = int(round(0.4 * lv * fs))                     # 최대 0.4초 지연
    return np.roll(y, s) if s else y.copy()


def deg_peak_broadening(y, fs, lv, rng):
    if lv <= 0:
        return y.copy()
    cutoff = max(1.2, 3.0 - 1.7 * lv)                 # 3.0 → 1.3 Hz
    nyq = 0.5 * fs
    b, a = butter(4, min(cutoff / nyq, 0.99), btype="low")
    return filtfilt(b, a, y)


def deg_quantization(y, fs, lv, rng):
    if lv <= 0:
        return y.copy()
    levels = max(3, int(round(64 * (1.0 - 0.95 * lv))))
    lo, hi = y.min(), y.max()
    if hi - lo < 1e-12:
        return y.copy()
    q = np.round((y - lo) / (hi - lo) * (levels - 1)) / (levels - 1)
    return q * (hi - lo) + lo


def deg_hr_bias(y, fs, lv, rng):
    """심박 자체를 틀리게 추정한 경우 — 시간축 압축으로 주파수 오차 유발."""
    n = len(y)
    factor = 1.0 + 0.5 * lv                            # 최대 +50% HR
    src = np.clip(np.arange(n) * factor, 0, n - 1)
    return np.interp(src, np.arange(n), y)


def deg_polarity_inversion(y, fs, lv, rng):
    """극성반전. y*cos(pi*lv): lv=0→그대로, lv=0.5→완전 소거(0), lv=1→완전반전."""
    return y * np.cos(np.pi * lv)


def _dominant_cardiac_freq(y, fs, band=CARDIAC_BAND):
    nperseg = min(256, len(y))
    if nperseg < 16:
        return None
    freqs, psd = welch(y, fs=fs, nperseg=nperseg)
    m = (freqs >= band[0]) & (freqs <= band[1])
    if not m.any() or psd[m].sum() <= 0:
        return None
    return float(freqs[m][np.argmax(psd[m])])


def deg_dicrotic_notch_attenuation(y, fs, lv, rng):
    """dicrotic notch(2차 고조파) 대역만 노치필터로 선택 감쇠."""
    if lv <= 0:
        return y.copy()
    f0 = _dominant_cardiac_freq(y, fs)
    if f0 is None:
        return y.copy()
    f_notch = 2.0 * f0
    nyq = 0.5 * fs
    bw = 0.3
    lo, hi = (f_notch - bw) / nyq, (f_notch + bw) / nyq
    if lo <= 1e-6 or hi >= 0.99 or lo >= hi:
        return y.copy()
    b, a = butter(2, [max(lo, 1e-6), min(hi, 0.99)], btype="bandstop")
    attenuated = filtfilt(b, a, y)
    return (1.0 - lv) * y + lv * attenuated


def deg_extra_beats(y, fs, lv, rng):
    """dropout(=missed beats)의 반대 방향. 가짜 추가 박동을 삽입."""
    if lv <= 0:
        return y.copy()
    z = y.copy()
    peaks, _ = find_peaks(y, distance=max(1, int(0.4 * fs)))
    if len(peaks) < 3:
        return z
    n_gaps = len(peaks) - 1
    n_insert = min(max(1, int(round(lv * n_gaps))), n_gaps)
    gap_idx = rng.choice(n_gaps, size=n_insert, replace=False)
    half = max(2, int(0.3 * fs))
    for g in gap_idx:
        p1, p2 = int(peaks[g]), int(peaks[g + 1])
        mid = (p1 + p2) // 2
        s, e = max(0, p1 - half), min(len(y), p1 + half)
        template = y[s:e] - np.median(y[s:e])
        ts = mid - (p1 - s)
        te = ts + len(template)
        if ts < 0 or te > len(z):
            continue
        z[ts:te] += template
    return z


def deg_peak_sharpening(y, fs, lv, rng):
    """peak_broadening의 반대 방향. unsharp masking으로 디테일을 과장."""
    if lv <= 0:
        return y.copy()
    nyq = 0.5 * fs
    cutoff = min(1.5 / nyq, 0.99)
    b, a = butter(4, cutoff, btype="low")
    smooth = filtfilt(b, a, y)
    detail = y - smooth
    return y + 3.0 * lv * detail


DEGRADATIONS = {
    "white_noise": deg_white_noise,
    "colored_noise": deg_colored_noise,
    "baseline_wander": deg_baseline_wander,
    "amplitude_scale": deg_amplitude_scale,
    "dc_offset": deg_dc_offset,
    "spike_burst": deg_spike_burst,
    "spike_dispersed": deg_spike_dispersed,
    "dropout": deg_dropout,
    "temporal_jitter": deg_temporal_jitter,
    "phase_shift": deg_phase_shift,
    "peak_broadening": deg_peak_broadening,
    "quantization": deg_quantization,
    "hr_bias": deg_hr_bias,
    "polarity_inversion": deg_polarity_inversion,
    "dicrotic_notch_attenuation": deg_dicrotic_notch_attenuation,
    "extra_beats": deg_extra_beats,
    "peak_sharpening": deg_peak_sharpening,
}


# ══════════════════════════════════════════════════════════════════
# 한 "가상 participant"의 컴포넌트 계산 (rpfi_eval과 동일 절차)
# ══════════════════════════════════════════════════════════════════

def components_for_case(ref_chunks, pred_chunks, fs, do_detrend, do_bpf, Lambda):
    """rpfi_eval.main()의 participant 처리와 완전히 동일한 절차.
    BWMD/SRE/WCR는 BPF 신호(L,P), EDD는 non-BPF 신호(L_e,P_e)를 쓴다
    (rpfi_eval.py의 preprocess_chunk_edd 설계와 동일한 이유)."""
    L = np.array([preprocess_chunk(c, fs, do_detrend, do_bpf, Lambda) for c in ref_chunks])
    P = np.array([preprocess_chunk(c, fs, do_detrend, do_bpf, Lambda) for c in pred_chunks])
    L_e = np.array([preprocess_chunk_edd(c, fs, do_detrend, Lambda) for c in ref_chunks])
    P_e = np.array([preprocess_chunk_edd(c, fs, do_detrend, Lambda) for c in pred_chunks])
    bwmd = float(np.mean([compute_bwmd(L[i], P[i], fs)[0] for i in range(len(L))]))
    y, p = L.ravel(), P.ravel()
    sre = compute_sre(y, p, fs)[0]
    wcr = compute_wcr(y, p)[0]
    edd = compute_edd(L_e.ravel(), P_e.ravel(), fs)
    return {"BWMD": bwmd, "SRE": sre, "WCR": wcr, "EDD": edd}


def make_case(fs, seg_len, n_chunks, rng, transform):
    """참조 chunk들을 만들고 transform으로 예측을 만든다."""
    ref, pred = [], []
    base, hr = synth_ppg(seg_len * n_chunks, fs, rng)
    for i in range(n_chunks):
        y = base[i * seg_len:(i + 1) * seg_len]
        ref.append(y)
        pred.append(transform(y, fs, rng, hr))
    return ref, pred


# ══════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="anchors")
    ap.add_argument("--fs-list", default="30,20", help="anchor를 통합할 fs 목록")
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--n-chunks", type=int, default=6, help="가상 participant당 chunk 수")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--levels", type=int, default=6)
    ap.add_argument("--q-lo", type=float, default=1.0, help="ideal 쪽 분위수(%)")
    ap.add_argument("--q-hi", type=float, default=99.0, help="null 쪽 분위수(%)")
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--derive-all", action="store_true",
                    help="WCR/EDD/SRE도 이론경계 대신 ideal↔null로 경험적 도출 "
                         "(비권장 — SRE는 과거 방식 재현/비교용으로만 사용할 것)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fs_list = [int(x) for x in args.fs_list.split(",")]
    do_detrend, do_bpf = not args.no_detrend, not args.no_bpf
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("RPFI anchor 재도출 + 열화 검증")
    print("=" * 78)
    print(f"  fs={fs_list}  chunk={args.eval_seconds}s × {args.n_chunks}개  reps={args.reps}")
    print(f"  전처리: detrend={do_detrend}  bandpass={do_bpf}  "
          f"(rpfi_eval.py 실행 옵션과 반드시 일치시킬 것)")
    print(f"  ※ rpfi_eval.py의 컴포넌트 함수를 직접 import → 척도 일치 보장")

    ensemble_rows, sweep_rows = [], []
    ideal_vals = defaultdict(list)
    null_vals = defaultdict(list)

    # ── ideal / null 앙상블 ──
    print("\n[1/2] 기준 앙상블 (ideal / null) 계산 중...")
    for fs in fs_list:
        seg_len = int(round(fs * args.eval_seconds))

        def t_ideal(y, fs_, r, hr):
            return y + 0.01 * np.std(y) * r.standard_normal(len(y))

        def t_null_ppg(y, fs_, r, hr):
            other = hr * r.uniform(0.6, 1.6)
            while abs(other - hr) < 8:
                other = hr * r.uniform(0.6, 1.6)
            return synth_ppg(len(y), fs_, r, hr_bpm=other)[0]

        def t_null_noise(y, fs_, r, hr):
            w = r.standard_normal(len(y))
            return _bp(w, fs_, 0.75, 3.0)

        for name, tf, store in [("ideal", t_ideal, ideal_vals),
                                ("null_ppg", t_null_ppg, null_vals),
                                ("null_noise", t_null_noise, null_vals)]:
            for rep in range(args.reps):
                ref, pred = make_case(fs, seg_len, args.n_chunks, rng, tf)
                comp = components_for_case(ref, pred, fs, do_detrend, do_bpf,
                                           args.lambda_detrend)
                for c in COMPONENTS:
                    store[c].append(comp[c])
                ensemble_rows.append(dict(ensemble=name, fs=fs, rep=rep, **comp))
        print(f"    fs={fs} 완료")

    # ── anchor 확정 ──
    # ★ 컴포넌트마다 anchor 근거가 다르다.
    #   BWMD       : 정의상 상한은 있으나(3.0) 실제로 도달하지 않는 느슨한 값이라
    #                (실측 anchor 이용률 92~100%, 클리핑 거의 0), ideal↔null 앙상블로
    #                "도달 가능한 최선 ↔ 무관한 신호" 구간을 잡는 편이 낫다.
    #   SRE        : 2026-07 anchor 재검토 전에는 BWMD와 같은 이유로 ideal↔null을
    #                썼으나(당시 anchor≈[0.0038, 1.4854]), 9개 run 실측에서 최대 35%
    #                클리핑(관측 최댓값 4.29)이 나와 "느슨해서 안 닿는다"는 전제가
    #                틀렸음이 확인됐다. RMSE는 z-norm 경로에서 [0,2]로 유계이고
    #                g(SNR)은 rpfi_eval.SRE_G_MAX로 명시적으로 클램프되므로
    #                SRE = RMSE·g(SNR) ≤ 2·SRE_G_MAX는 평가 대상 모델 풀과 무관하게
    #                항상 성립하는 **닫힌 형태 이론상한**이다. 이후로는 SRE도 이론경계를
    #                직접 쓴다(BWMD보다 오히려 더 강한 pool-independence 근거).
    #   WCR / EDD  : 정의상 이미 [0,1]로 유계이므로 **이론 경계**를 그대로 쓴다.
    #                이 둘에 ideal↔null을 적용하면 오히려 왜곡된다:
    #                  · WCR은 null에서 음수(-0.01)가 나와 하한이 음수로 잡힌다.
    #                  · EDD는 null(무관 맥파)의 잔차가 두 사인파의 차라서 오히려
    #                    가우시안에 가까워 값이 낮게 나온다. 즉 EDD는 ideal→null 축을
    #                    따라 단조롭지 않아 이 방식으로 anchor를 잡을 수 없다.
    THEORETICAL = {"WCR": (0.0, 1.0), "EDD": (0.0, 1.0), "SRE": (0.0, 2.0 * SRE_G_MAX)}

    anchors, anchor_basis = {}, {}
    for c in COMPONENTS:
        if c in THEORETICAL and not args.derive_all:
            anchors[c] = THEORETICAL[c]
            anchor_basis[c] = f"theoretical [{THEORETICAL[c][0]:g}, {THEORETICAL[c][1]:g}]"
            continue
        iv = np.asarray(ideal_vals[c])
        nv = np.asarray(null_vals[c])
        if DIRECTION[c] == "lower":
            lo = float(np.percentile(iv, args.q_lo))
            hi = float(np.percentile(nv, args.q_hi))
        else:
            hi = float(np.percentile(iv, args.q_hi))
            lo = float(np.percentile(nv, args.q_lo))
        if hi <= lo:                       # 안전장치
            hi = lo + 1e-6
        anchors[c] = (lo, hi)
        anchor_basis[c] = f"ideal p{args.q_lo:g} ↔ null p{args.q_hi:g}"

    print("\n" + "=" * 78)
    print("재도출된 anchor")
    print("=" * 78)
    print(f"  {'Comp':>6s} {'dir':>7s} | {'ideal(중앙)':>12s} {'null(중앙)':>12s} | "
          f"{'anchor_lo':>11s} {'anchor_hi':>11s} | 근거")
    for c in COMPONENTS:
        lo, hi = anchors[c]
        print(f"  {c:>6s} {DIRECTION[c]:>7s} | "
              f"{np.median(ideal_vals[c]):>12.5f} {np.median(null_vals[c]):>12.5f} | "
              f"{lo:>11.5f} {hi:>11.5f} | {anchor_basis[c]}")
    print("\n  ideal = 잔차가 신호의 1%인 사실상 완벽한 재구성 (RPFI 100의 조작적 정의)")
    print("  null  = 심박이 다른 독립 맥파 + 대역통과 백색잡음 (RPFI 0의 조작적 정의)")

    # ── 열화 스윕 (R1-8) ──
    total = len(DEGRADATIONS) * args.levels * args.reps * len(fs_list)
    print(f"\n[2/2] 열화 스윕 {len(DEGRADATIONS)}종 × {args.levels}단계 × "
          f"{args.reps}회 × fs{len(fs_list)}종 = {total} 케이스...")
    done = 0
    for fs in fs_list:
        seg_len = int(round(fs * args.eval_seconds))
        for dname, dfn in DEGRADATIONS.items():
            for li in range(args.levels):
                lv = li / (args.levels - 1) if args.levels > 1 else 1.0
                for rep in range(args.reps):
                    def tf(y, fs_, r, hr, _f=dfn, _lv=lv):
                        return _f(y, fs_, _lv, r)
                    ref, pred = make_case(fs, seg_len, args.n_chunks, rng, tf)
                    comp = components_for_case(ref, pred, fs, do_detrend, do_bpf,
                                               args.lambda_detrend)
                    sweep_rows.append(dict(degradation=dname, fs=fs, level=lv,
                                           rep=rep, **comp))
                    done += 1
        print(f"    fs={fs} 완료 ({done}/{total})")

    # ── 단조성 / 민감도 리포트 ──
    lines = []
    lines.append("=" * 96)
    lines.append("열화 검증: 각 컴포넌트가 강도 증가에 단조 반응하는가 (R1-8)")
    lines.append("=" * 96)
    lines.append("  ρ = level과 컴포넌트값의 Spearman 상관 (부호는 '나빠지는 방향'으로 정렬)")
    lines.append("  Δ/range = (최대강도 중앙값 − 무열화 중앙값)을 anchor 범위로 나눈 비율")
    lines.append("")
    header = f"  {'degradation':<20s}" + "".join(
        f"{c + ' ρ':>10s}{c + ' Δ/r':>10s}" for c in COMPONENTS)
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    by_deg = defaultdict(lambda: defaultdict(list))
    for r in sweep_rows:
        for c in COMPONENTS:
            by_deg[r["degradation"]][c].append((r["level"], r[c]))

    insensitive = []
    for dname in DEGRADATIONS:
        cells = ""
        for c in COMPONENTS:
            pts = by_deg[dname][c]
            lv = np.array([p[0] for p in pts]); vals = np.array([p[1] for p in pts])
            rho, _ = spearmanr(lv, vals)
            rho = 0.0 if np.isnan(rho) else rho
            if DIRECTION[c] == "higher":
                rho = -rho          # 나빠지는 방향을 +로 통일
            base = np.median(vals[lv == lv.min()])
            worst = np.median(vals[lv == lv.max()])
            delta = (worst - base) if DIRECTION[c] == "lower" else (base - worst)
            lo, hi = anchors[c]
            frac = delta / (hi - lo)
            cells += f"{rho:>10.3f}{frac:>10.3f}"
            if abs(frac) < 0.02:
                insensitive.append((dname, c, frac))
        lines.append(f"  {dname:<20s}{cells}")

    lines.append("")
    lines.append("  해석: ρ>0 이면 '열화가 심할수록 지표가 나빠진다'(정상).")
    lines.append("        Δ/range 는 anchor 스케일 대비 반응 폭. 0.02 미만이면 사실상 무반응.")
    lines.append("")
    if insensitive:
        lines.append("  ⚠ 사실상 무반응인 (열화, 컴포넌트) 쌍:")
        for d, c, f in insensitive:
            lines.append(f"      {d:<20s} {c:<6s} Δ/range={f:+.4f}")
        lines.append("")
        lines.append("    amplitude_scale / dc_offset 가 여기 있으면 정상입니다 —")
        lines.append("    평가경로의 z-normalize가 진폭·오프셋을 제거하기 때문이며,")
        lines.append("    이는 진폭 캘리브레이션을 RPFI 밖 별도 진단으로 분리한 설계와 일치합니다(R1-4).")
        lines.append("    그 외 항목이 있다면 해당 컴포넌트의 특이성을 Discussion에서 다뤄야 합니다.")

    non_monotonic = []
    for dname in DEGRADATIONS:
        for c in COMPONENTS:
            pts = by_deg[dname][c]
            lv = np.array([p[0] for p in pts]); vals = np.array([p[1] for p in pts])
            rho, _ = spearmanr(lv, vals)
            rho = 0.0 if np.isnan(rho) else rho
            if DIRECTION[c] == "higher":
                rho = -rho
            if rho < -0.3:
                non_monotonic.append((dname, c, rho))
    if non_monotonic:
        lines.append("")
        lines.append("  ⚠ 역방향 반응(열화가 심해질수록 지표가 좋아짐) — 반드시 확인:")
        for d, c, r in non_monotonic:
            lines.append(f"      {d:<20s} {c:<6s} ρ={r:+.3f}")

    # ── burst vs dispersed 구분력 통계검정 (R1-8/M10, "동일 크기 잔차라도
    #    burst냐 dispersed냐를 EDD가 구분하는가") — 최대 강도(lv=1.0 → 정확히
    #    5% spike fraction, deg_spike_burst/dispersed의 k=0.05*lv*len(y) 정의상)
    #    에서 spike_burst와 spike_dispersed의 EDD 분포를 rep×fs 전체 풀링해 비교.
    lines.append("")
    lines.append("=" * 96)
    lines.append("burst vs dispersed 구분력 통계검정 (최대강도=5% spike fraction, EDD 기준)")
    lines.append("=" * 96)
    max_level = 1.0  # lv=li/(levels-1)의 최댓값 — deg_spike_*의 k=0.05*lv*len(y) 정의상 정확히 5%
    burst_edd = np.array([r["EDD"] for r in sweep_rows
                          if r["degradation"] == "spike_burst"
                          and abs(r["level"] - max_level) < 1e-9])
    disp_edd = np.array([r["EDD"] for r in sweep_rows
                         if r["degradation"] == "spike_dispersed"
                         and abs(r["level"] - max_level) < 1e-9])
    lines.append(f"  n(burst)={len(burst_edd)}  n(dispersed)={len(disp_edd)}  "
                f"(reps={args.reps} × fs{len(fs_list)}종 = {args.reps*len(fs_list)}이어야 정상)")
    if len(burst_edd) >= 2 and len(disp_edd) >= 2:
        lines.append(f"  burst      EDD: mean={burst_edd.mean():.4f}  std={burst_edd.std():.4f}  "
                    f"median={np.median(burst_edd):.4f}")
        lines.append(f"  dispersed  EDD: mean={disp_edd.mean():.4f}  std={disp_edd.std():.4f}  "
                    f"median={np.median(disp_edd):.4f}")
        t_stat, p_ttest = ttest_ind(burst_edd, disp_edd, equal_var=False)
        u_stat, p_mwu = mannwhitneyu(burst_edd, disp_edd, alternative="two-sided")
        lines.append(f"  Welch t-test:      t={t_stat:.4f}  p={p_ttest:.6g}")
        lines.append(f"  Mann-Whitney U:    U={u_stat:.4f}  p={p_mwu:.6g}")
        lines.append(f"  → {'통계적으로 유의하게 구분됨' if p_mwu < 0.05 else '유의한 차이 없음(⚠ 재검토 필요)'} "
                    f"(Mann-Whitney 기준, α=0.05)")
    else:
        lines.append("  ⚠ 표본 부족 — --reps를 늘려서 재실행 필요")

    report = "\n".join(lines)
    print("\n" + report)

    # ── 저장 ──
    json.dump({c: list(anchors[c]) for c in COMPONENTS},
              open(out / "rpfi_anchors.json", "w"), indent=2)
    with open(out / "anchor_ensembles.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ensemble_rows[0].keys()))
        w.writeheader(); w.writerows(ensemble_rows)
    with open(out / "degradation_sweep.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
        w.writeheader(); w.writerows(sweep_rows)
    meta = dict(fs_list=fs_list, eval_seconds=args.eval_seconds,
                n_chunks=args.n_chunks, reps=args.reps, levels=args.levels,
                detrend=do_detrend, bandpass=do_bpf, lambda_detrend=args.lambda_detrend,
                q_lo=args.q_lo, q_hi=args.q_hi, seed=args.seed,
                derive_all=args.derive_all, anchor_basis=anchor_basis,
                anchor_definition="BWMD: ideal(1% residual) vs null(unrelated PPG + "
                                  "bandpassed noise), p{q_lo}/p{q_hi} of that ensemble. "
                                  "SRE: theoretical [0, 2*SRE_G_MAX] (RMSE bounded to [0,2] "
                                  "on z-normalized signals; g(SNR) clamped to "
                                  "[SRE_G_MIN, SRE_G_MAX] in rpfi_eval.py). "
                                  "WCR/EDD: theoretical [0,1].".format(
                                      q_lo=args.q_lo, q_hi=args.q_hi))
    json.dump(meta, open(out / "anchor_meta.json", "w"), indent=2)
    (out / "degradation_report.txt").write_text(report, encoding="utf-8")

    print("\n저장:")
    for f in ["rpfi_anchors.json", "anchor_ensembles.csv",
              "degradation_sweep.csv", "degradation_report.txt", "anchor_meta.json"]:
        print(f"  {out / f}")
    print(f"\n다음 실행:")
    print(f"  python rpfi_eval.py --dataset ubfc --runs runs/ubfc_cv5 \\")
    print(f"      --label \"../real data/UBFC_label.npy\" \\")
    print(f"      --anchors {out / 'rpfi_anchors.json'} --out results_eval")


if __name__ == "__main__":
    main()

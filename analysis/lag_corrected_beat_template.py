"""
lag_corrected_beat_template.py — lag보정 + 진폭정규화 비트 템플릿 비교 (M4 대응)
================================================================================
리뷰어 M4: "A more defensible approach would compare lag-corrected and
amplitude-normalized beat templates for morphology, while separately
evaluating beat timing." 그리고 "residual lag distribution, ... polarity
correction, offset removal, amplitude-calibration procedure"를 보고하라는 요구.

BWMD(DTW 기반)는 이미 국소적 시간왜곡을 허용하지만, 리뷰어는 그와 별개로 더
단순하고 투명한 방법을 요구한다:
  1) 전역 lag 하나를 cross-correlation으로 추정
  2) 필요하면 극성(±1) 보정
  3) 그 lag/극성으로 정렬
  4) label 기준 피크로 비트를 잘라 각 비트를 자기 진폭으로 정규화
  5) 비트 평균 = 템플릿, label 템플릿 vs pred 템플릿 비교

"beat timing"과 "morphology"를 리뷰어 요구대로 분리해서 보고한다:
  - 타이밍 증거 = lag_sec의 분포 (이 스크립트가 새로 만듦)
  - 형태 증거   = template_corr/template_rmse (lag를 이미 제거한 뒤 비교하므로
                  타이밍 문제가 형태 비교에 섞이지 않음)
BWMD의 PeakDiff/HR MAE/PRV는 이미 타이밍을 별도로 다루고 있으므로 이 스크립트는
그것들과 겹치지 않고, BWMD와는 **독립적인 방법으로 같은 결론이 나오는지 교차검증**
하는 역할을 한다.

전처리는 rpfi_eval.load_participant_data가 만든 participant별 (label, pred) 청크
배열을 그대로 재사용한다(이미 detrend+BPF+z-norm 적용됨, 전처리 파이프라인 이원화 방지).

사용법
------
    python lag_corrected_beat_template.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" --models all \
        --results-root . --agg geometric \
        --out lag_template_ubfc
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, find_peaks
from scipy.stats import pearsonr, skew, spearmanr

try:
    import rpfi_eval as R
except ImportError as e:
    sys.exit(f"[오류] rpfi_eval.py를 import할 수 없습니다 ({e}). 같은 폴더에서 실행하세요.")

ALL_MODELS = ["pos", "rnn", "lstm", "bilstm", "physdiff", "transformer",
              "mamba", "bimamba", "bimamba_blend"]


# ══════════════════════════════════════════════════════════════
# 핵심: lag/극성 추정 → 정렬 → 비트 추출/정규화 → 템플릿 비교
# ══════════════════════════════════════════════════════════════

def _search_lag(y, yhat, fs, max_lag_sec, polarities):
    """FFT cross-correlation으로 주어진 극성 후보들 중 최적 (lag, polarity)를 찾는다.
    correlation_lags의 peak lag는 apply_lag_polarity가 기대하는 부호와 반대라서
    (실측 확인됨) 여기서 부호를 뒤집어 반환한다."""
    max_lag = int(round(max_lag_sec * fs))
    y0 = y - y.mean()
    best = None  # (corr_val, lag_samples_intuitive, polarity)
    for polarity in polarities:
        yh0 = polarity * (yhat - yhat.mean())
        c = correlate(y0, yh0, mode="full", method="fft")
        lags = correlation_lags(len(y0), len(yh0), mode="full")
        mask = np.abs(lags) <= max_lag
        c_m, lags_m = c[mask], lags[mask]
        if len(c_m) == 0:
            continue
        idx = int(np.argmax(c_m))
        norm = np.sqrt(np.sum(y0 ** 2) * np.sum(yh0 ** 2)) + 1e-12
        corr_val = float(c_m[idx] / norm)
        lag_samples_intuitive = -int(lags_m[idx])
        if best is None or corr_val > best[0]:
            best = (corr_val, lag_samples_intuitive, polarity)
    corr_val, lag_samples, polarity = best
    return lag_samples / fs, corr_val, polarity


def find_lag_and_polarity(y, yhat, fs, max_lag_sec=1.0, skew_threshold=0.05):
    """극성은 skewness(왜도) 부호로 결정하고, 지연은 그렇게 고정한 극성으로
    넓은 범위에서 cross-correlation 탐색한다.

    ★★★ 중요: 좁은창 2단계 탐색(이전 버전)은 실패했다 ★★★
    처음엔 "아주 좁은 창(±0.15초)에서 극성만 정하고 넓은 창에서 지연을 찾는다"는
    방식을 썼는데, 실제 lag가 0.15초를 넘는 경우(실측상 흔함 — UBFC도 median_abs
    lag가 모델별로 0.27초까지, COHFACE는 0.5~0.7초)에는 좁은 창 안에 진짜 정답이
    아예 없어서 노이즈만 보고 극성을 잘못 판정하는 문제가 있었다(회귀테스트로 재현
    확인함).

    근본 원인: BPF된 심박 신호처럼 협대역 준정현파에 가까운 신호는, 이동 범위를
    충분히 넓게 주면 '극성반전+반주기 이동'과 '무반전+소폭 이동'의 상관계수가
    원천적으로 구별 불가능하다(주기가 있는 한 이동 자유도만으로 사실상 언제나
    거의 완벽한 상관을 만들 수 있음) — 즉 교차상관 기반 결합(극성,지연) 탐색은
    창 크기와 무관하게 근본적으로 불안정하다.

    해결: 이동에 대해 불변인 특징(skewness)으로 극성을 **독립적으로** 정한다.
    실측 검증(진짜 np.roll 시간이동, BPF 후): skewness 부호는 이동량과 무관하게
    거의 완벽히 안정적(예: -0.45 근방에서 이동량에 따라 ±0.03 이내 변동, 노이즈가
    있어도 부호는 안 바뀜). 극성이 이렇게 고정되면 지연 탐색은 더 이상 극성과
    결합된 문제가 아니라 단일 극성 안에서의 통상적인 cross-correlation 최댓값
    탐색이 되어 안정적이다.
    """
    if np.std(y) < 1e-8 or np.std(yhat) < 1e-8:
        polarity = 1
    else:
        sk_y, sk_yhat = float(skew(y)), float(skew(yhat))
        if abs(sk_y) > skew_threshold and abs(sk_yhat) > skew_threshold:
            polarity = 1 if np.sign(sk_y) == np.sign(sk_yhat) else -1
        else:
            polarity = 1   # 둘 다 거의 대칭이면 판단 근거가 약하므로 '반전 없음'을 기본값으로
    lag_sec, corr_val, _ = _search_lag(y, yhat, fs, max_lag_sec, (polarity,))

    # ★ 추가 모호성: max_lag_sec이 카디악 주기보다 넓으면 '정답보다 정수 주기만큼
    # 통째로 어긋난' lag도 상관계수가 사실상 동일해서(주기신호라 당연함) 노이즈로
    # 잘못 선택될 수 있다(회귀테스트로 재현: 참값 +0.4초인데 -0.433초로 나옴 —
    # 둘의 차이 0.833초가 정확히 카디악 주기와 일치). 추정 국소 주기의 절반
    # 이내로 lag를 접어넣어(modular wrap) 가장 작은 등가 lag로 정규화한다.
    # 부수 효과: apply_lag_polarity가 잘라내는 경계 구간도 줄어들어 비트 통계가 좋아짐.
    py = peak_indices(y, fs)
    if len(py) >= 2:
        period_sec = float(np.median(np.diff(py))) / fs
        if period_sec > 0:
            lag_sec = ((lag_sec + period_sec / 2) % period_sec) - period_sec / 2

    return lag_sec, polarity, corr_val


def apply_lag_polarity(y, yhat, lag_samples, polarity):
    """yhat에 극성 적용 후 lag_samples만큼 시프트, 겹치는 유효구간만 반환.
    lag_samples > 0: yhat이 y보다 lag_samples만큼 뒤처져 있었다는 뜻(양의 지연)."""
    yh = polarity * yhat
    if lag_samples > 0:
        yh2 = yh[lag_samples:]
        y2 = y[:len(yh2)]
    elif lag_samples < 0:
        y2 = y[-lag_samples:]
        yh2 = yh[:len(y2)]
    else:
        y2, yh2 = y.copy(), yh.copy()
    n = min(len(y2), len(yh2))
    return y2[:n], yh2[:n]


def peak_indices(sig, fs, max_hr=150):
    """rpfi_eval.peak_times와 동일한 distance 기준(0.4s), 샘플 인덱스로 반환."""
    peaks, _ = find_peaks(sig, distance=max(1, int(0.4 * fs)))
    return peaks


def extract_beats(sig, peaks, fs, half_sec=0.4):
    """피크 중심 고정길이로 잘라 각 비트를 자기 진폭(peak-to-peak)으로 정규화."""
    half = int(round(half_sec * fs))
    beats = []
    for p in peaks:
        s, e = p - half, p + half
        if s < 0 or e > len(sig):
            continue
        seg = sig[s:e].copy()
        amp = np.max(seg) - np.min(seg)
        if amp < 1e-8:
            continue
        beats.append(seg / amp)
    return beats


def beat_template(sig, peaks, fs, half_sec=0.4, min_beats=3):
    beats = extract_beats(sig, peaks, fs, half_sec)
    if len(beats) < min_beats:
        return None, len(beats)
    return np.mean(beats, axis=0), len(beats)


def process_chunk_pair(y_chunk, yhat_chunk, fs, max_lag_sec=1.0, half_sec=0.4, min_beats=3):
    lag_sec, polarity, corr_at_lag = find_lag_and_polarity(y_chunk, yhat_chunk, fs, max_lag_sec)
    lag_samples = int(round(lag_sec * fs))
    y_al, yh_al = apply_lag_polarity(y_chunk, yhat_chunk, lag_samples, polarity)
    if len(y_al) < fs * 2:
        return None
    peaks = peak_indices(y_al, fs)              # label 기준 피크(정렬 이후 = 두 신호에 공통 적용)
    tmpl_y, n_beats_y = beat_template(y_al, peaks, fs, half_sec, min_beats)
    tmpl_p, n_beats_p = beat_template(yh_al, peaks, fs, half_sec, min_beats)
    if tmpl_y is None or tmpl_p is None:
        return None
    if np.std(tmpl_y) < 1e-8 or np.std(tmpl_p) < 1e-8:
        template_corr = 0.0
    else:
        template_corr = float(np.corrcoef(tmpl_y, tmpl_p)[0, 1])
    template_rmse = float(np.sqrt(np.mean((tmpl_y - tmpl_p) ** 2)))
    return dict(lag_sec=lag_sec, polarity_flipped=(polarity == -1),
               corr_at_lag=corr_at_lag, n_beats=min(n_beats_y, n_beats_p),
               template_corr=template_corr, template_rmse=template_rmse)


# ══════════════════════════════════════════════════════════════
# 모델 1개 전체 실행
# ══════════════════════════════════════════════════════════════

def run_model(runs_dir, label_full, model, fs, seg_len, do_detrend, do_bpf, Lambda,
             max_lag_sec, half_sec, min_beats,
             n_participants=None, n_chunks_per_participant=None, seed=0):
    data = R.load_participant_data(runs_dir, label_full, model, fs, seg_len,
                                   do_detrend, do_bpf, Lambda)
    if not data:
        return []
    rng = np.random.default_rng(seed)
    subs = sorted(data)
    if n_participants and len(subs) > n_participants:
        subs = sorted(rng.choice(subs, n_participants, replace=False).tolist())

    rows = []
    for subj in subs:
        d = data[subj]
        L, P = d["label"], d["pred"]
        n_chunk = len(L)
        if n_chunk == 0:
            continue
        idx = (sorted(rng.choice(n_chunk, min(n_chunks_per_participant, n_chunk),
                                 replace=False).tolist())
              if n_chunks_per_participant else list(range(n_chunk)))
        for c in idx:
            r = process_chunk_pair(L[c], P[c], fs, max_lag_sec, half_sec, min_beats)
            if r is None:
                continue
            r.update(model=model, participant=subj, chunk=int(c))
            rows.append(r)
    return rows


# ══════════════════════════════════════════════════════════════
# 기존 RPFI/BWMD와 교차검증 (선택)
# ══════════════════════════════════════════════════════════════

def find_col(df, keywords):
    for kw in keywords:
        for c in df.columns:
            if c.lower() == kw.lower():
                return c
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None


def cross_validate_with_rpfi(df, results_root, dataset, agg, out_dir):
    d = Path(results_root) / f"results_{dataset}_final"
    cands = sorted(d.glob(f"rpfi_participant_{dataset}_{agg}*.csv"))
    if not cands:
        print(f"[교차검증 건너뜀] {d}에 rpfi_participant_*.csv 없음")
        return None
    rpfi_df = pd.read_csv(cands[0])
    model_col = find_col(rpfi_df, ["model"])
    subj_col = find_col(rpfi_df, ["participant", "subject"])
    rpfi_col = find_col(rpfi_df, ["rpfi"])
    bwmd_col = find_col(rpfi_df, ["bwmd"])
    if not (model_col and subj_col and rpfi_col):
        print(f"[교차검증 건너뜀] 컬럼 자동탐지 실패: {list(rpfi_df.columns)}")
        return None

    agg_here = (df.groupby(["model", "participant"])
               .agg(template_corr=("template_corr", "mean"),
                    template_rmse=("template_rmse", "mean"),
                    lag_sec=("lag_sec", "mean")).reset_index())
    r = rpfi_df.rename(columns={model_col: "model", subj_col: "participant",
                                rpfi_col: "RPFI", bwmd_col: "BWMD"})
    r["model"] = r["model"].astype(str).str.lower()
    agg_here["model"] = agg_here["model"].astype(str).str.lower()
    merged = pd.merge(agg_here, r[["model", "participant", "RPFI", "BWMD"]],
                      on=["model", "participant"], how="inner")
    if merged.empty:
        print("[교차검증] 조인 결과 0행 — participant ID 표기 확인 필요")
        return None

    print(f"\n{'='*70}\n독립적 방법(lag보정+진폭정규화 템플릿) vs 기존 RPFI/BWMD 교차검증\n{'='*70}")
    print(f"조인된 participant-model: {len(merged)}건")
    for tgt in ["RPFI", "BWMD"]:
        rho, p = spearmanr(merged["template_corr"], merged[tgt])
        direction = "양의 상관 기대(template_corr↑=형태일치, RPFI↑=충실도)" if tgt == "RPFI" \
            else "음의 상관 기대(template_corr↑=형태일치, BWMD↓=거리이므로 좋음)"
        print(f"  template_corr vs {tgt}: Spearman ρ={rho:.3f} (p={p:.2e})  [{direction}]")
    merged.to_csv(Path(out_dir) / f"lag_template_vs_rpfi_crosscheck_{dataset}.csv", index=False)
    return merged


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--models", default="all")
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--max-lag-sec", type=float, default=1.0,
                    help="lag 탐색 범위(±초). 생리적으로 pulse transit time은 보통 1초 미만")
    ap.add_argument("--half-sec", type=float, default=0.4, help="비트 추출 반폭(초)")
    ap.add_argument("--min-beats", type=int, default=3, help="템플릿 최소 비트 수")
    ap.add_argument("--n-participants", type=int, default=0, help="속도용 서브샘플, 0=전체")
    ap.add_argument("--n-chunks-per-participant", type=int, default=0, help="0=전체 청크")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results-root", default=None,
                    help="지정하면 results_{dataset}_final/의 기존 RPFI와 교차검증")
    ap.add_argument("--agg", default="geometric")
    ap.add_argument("--out", default="lag_template_report")
    args = ap.parse_args()

    fs = R.FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    do_detrend, do_bpf = not args.no_detrend, not args.no_bpf
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
             if args.models != "all" else R.discover_models(args.runs))
    if not models:
        raise SystemExit(f"{args.runs}에서 모델을 찾지 못했습니다.")
    print(f"[모델] {models}")

    label_full = np.load(args.label)
    all_rows = []
    for model in models:
        print(f"  -- {model} 처리 중...")
        rows = run_model(args.runs, label_full, model, fs, seg_len, do_detrend, do_bpf,
                         args.lambda_detrend, args.max_lag_sec, args.half_sec, args.min_beats,
                         args.n_participants or None, args.n_chunks_per_participant or None,
                         args.seed)
        print(f"     청크 {len(rows)}개에서 템플릿 계산 완료")
        all_rows.extend(rows)

    if not all_rows:
        raise SystemExit("결과가 없습니다. 경로/모델명을 확인하세요.")

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / f"lag_template_chunklevel_{args.dataset}.csv", index=False)

    print(f"\n{'='*70}\n(1) Lag 분포 — '별도로 평가하는 beat timing' 증거\n{'='*70}")
    lag_summary = df.groupby("model")["lag_sec"].agg(["mean", "std", "median",
                                                       lambda x: x.abs().median()])
    lag_summary.columns = ["mean_sec", "std_sec", "median_sec", "median_abs_sec"]
    print(lag_summary.to_string())
    lag_summary.to_csv(out_dir / f"lag_distribution_{args.dataset}.csv")

    flip_rate = df.groupby("model")["polarity_flipped"].mean()
    print(f"\n극성 반전이 필요했던 청크 비율(모델별):")
    print(flip_rate.to_string())
    if (flip_rate > 0.05).any():
        print("  ⚠ 5% 넘게 극성반전이 필요한 모델이 있음 — 부호 관례 불일치 의심, 원본 코드 확인 권장")

    print(f"\n{'='*70}\n(2) Lag보정+진폭정규화 템플릿 상관/RMSE — 'morphology' 증거\n{'='*70}")
    tmpl_summary = df.groupby("model")[["template_corr", "template_rmse", "n_beats"]].mean()
    tmpl_summary = tmpl_summary.sort_values("template_corr", ascending=False)
    print(tmpl_summary.to_string())
    tmpl_summary.to_csv(out_dir / f"template_summary_{args.dataset}.csv")

    if args.results_root:
        cross_validate_with_rpfi(df, args.results_root, args.dataset, args.agg, out_dir)

    print(f"\n모든 산출물 저장: {out_dir}/")
    print("\n[response letter에 쓸 수 있는 구조]")
    print("  - '동기화 절차' 문단: lag_sec 분포(평균/중앙값/표준편차)를 그대로 보고")
    print("    (리뷰어가 요구한 'residual lag distribution'에 직접 대응)")
    print("  - '극성/오프셋' 문단: 극성반전 비율 보고 (대부분 0%에 가까우면 부호 관례가")
    print("    일관됨을 보여주는 근거) + preprocess_chunk의 detrend가 오프셋 제거를 이미 수행함을 명시")
    print("  - '형태 vs 타이밍 분리 검증' 문단: template_corr(형태, lag 제거 후)와")
    print("    BWMD/RPFI(형태+타이밍 통합)가 같은 방향으로 움직이면 -> 독립적 방법 간 수렴,")
    print("    BWMD가 아티팩트가 아니라는 교차검증 증거로 사용")


if __name__ == "__main__":
    main()

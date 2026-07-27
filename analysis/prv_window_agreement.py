"""
prv_window_agreement.py — 10초 청크 PRV vs 긴 윈도우(recording 전체) PRV 일치도 (M5 대응)
================================================================================
리뷰어 M5: "Please repeat the analysis using substantially longer windows where
available, report the number of detected beats per window, and quantify
agreement between 10-second and longer-duration SDNN/RMSSD."

라벨(contact PPG) **하나만으로** 검증한다 — 이건 "모델이 예측을 잘 하는가"가 아니라
"10초 청크로 SDNN/RMSSD를 재는 방법 자체가 믿을만한가"를 묻는 방법론 검증이라
예측값이 필요 없다.

recording 경계를 넘으면 가짜 RR이 생기므로(rpfi_eval.py의 prv_from_chunks 설계와
동일 이유) 반드시 **같은 recording 안에서** "짧은 청크 여러 개"와 "그 recording
전체(긴 윈도우)"를 비교한다. UBFC/PURE/COHFACE 전부 recording 길이가 대략
50~65초라(각각 평균 57.75s/64s/60.5s, mapping 데이터 기반 추정) 10초 청크 대비
5~6배 긴 창을 확보할 수 있다.

절차 (recording 단위)
----------------------
1) "짧음" = rpfi_eval.py가 실제로 보고하는 값과 동일한 방법: 10초
   non-overlapping 청크로 잘라 각각 preprocess_chunk 후 prv_from_chunks
2) "긺" = 같은 길이(청크 개수 × 10초, 자투리 제외)를 통째로 preprocess_chunk
   한 뒤 단일 청크로 prv_from_chunks
3) 두 값을 Bland-Altman(bias, 95% LoA) + Pearson/Spearman + MAE로 비교

recording 목록 복원: runs/{dataset}_cv{K}/pos_f{k}_info.json (k=0..K-1)의
test_ranges를 fold 전체에 걸쳐 합친다 — POS는 학습이 없어 모든 fold에 존재하고,
recording 경계 자체는 어떤 모델을 골라도 동일하므로(라벨만 쓰기 때문) 재사용.

사용법
------
    python prv_window_agreement.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" --out prv_agreement_ubfc
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

try:
    import rpfi_eval as R
except ImportError as e:
    sys.exit(f"[오류] rpfi_eval.py를 import할 수 없습니다 ({e}). "
             f"같은 폴더에서 실행하세요.")


def collect_all_recordings(runs_dir, n_folds, model="pos"):
    """지정 모델의 전 fold info.json에서 test_ranges를 합쳐 데이터셋 전체
    recording 목록(구간, participant)을 복원한다."""
    root = Path(runs_dir)
    ranges, subj_of = [], []
    found_folds = []
    for k in range(n_folds):
        f = root / f"{model}_f{k}_info.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text())
        ranges.extend(tuple(r) for r in info["test_ranges"])
        subj_of.extend(info["test_subject_of_range"])
        found_folds.append(k)
    if not found_folds:
        raise FileNotFoundError(
            f"{root}/{model}_f*_info.json 을 하나도 못 찾았습니다. "
            f"--model 이나 --runs 경로를 확인하세요.")
    return ranges, subj_of, found_folds


def compare_recording(y_raw, fs, seg_len, do_detrend, do_bpf, Lambda, min_chunks=3):
    n = (len(y_raw) // seg_len) * seg_len
    n_chunks = n // seg_len
    if n_chunks < min_chunks:
        return None
    y = y_raw[:n]

    # 짧음: rpfi_eval.py와 완전히 동일한 절차 — 청크별 전처리 후 prv_from_chunks
    short_chunks = [R.preprocess_chunk(y[c * seg_len:(c + 1) * seg_len], fs,
                                       do_detrend, do_bpf, Lambda)
                    for c in range(n_chunks)]
    prv_short = R.prv_from_chunks(short_chunks, fs)

    # 긺: 동일한 길이(n_chunks*seg_len, 자투리 제외로 공정 비교)를 통째로 전처리
    long_sig = R.preprocess_chunk(y, fs, do_detrend, do_bpf, Lambda)
    prv_long = R.prv_from_chunks([long_sig], fs)

    return dict(n_chunks=n_chunks, duration_sec=n / fs,
               sdnn_short=prv_short["sdnn_ms"], sdnn_long=prv_long["sdnn_ms"],
               rmssd_short=prv_short["rmssd_ms"], rmssd_long=prv_long["rmssd_ms"],
               hr_short=prv_short["mean_hr_bpm"], hr_long=prv_long["mean_hr_bpm"],
               n_rr_short=prv_short["n_rr"], n_rr_long=prv_long["n_rr"])


def bland_altman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 3:
        return dict(n=len(a), bias=np.nan, sd_diff=np.nan, loa_lo=np.nan, loa_hi=np.nan,
                    pearson=np.nan, spearman=np.nan, mae=np.nan)
    diff = a - b
    bias, sd = float(diff.mean()), float(diff.std(ddof=1)) if len(a) > 1 else 0.0
    r, _ = pearsonr(a, b) if len(a) > 1 else (np.nan, np.nan)
    rho, _ = spearmanr(a, b) if len(a) > 1 else (np.nan, np.nan)
    return dict(n=len(a), bias=bias, sd_diff=sd,
               loa_lo=bias - 1.96 * sd, loa_hi=bias + 1.96 * sd,
               pearson=float(r) if not np.isnan(r) else np.nan,
               spearman=float(rho) if not np.isnan(rho) else np.nan,
               mae=float(np.mean(np.abs(diff))))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True, help="runs/{dataset}_cv{K} 디렉토리")
    ap.add_argument("--label", required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--model", default="pos",
                    help="recording 경계 복원용 모델(라벨만 쓰므로 아무 모델이나 무방, "
                        "전 fold에 존재하는 pos가 기본값)")
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--min-chunks", type=int, default=3,
                    help="recording당 최소 청크 수(=최소 30초). 이보다 짧은 recording은 제외")
    ap.add_argument("--out", default="prv_window_agreement_report")
    args = ap.parse_args()

    fs = R.FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    do_detrend, do_bpf = not args.no_detrend, not args.no_bpf
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    label = np.load(args.label)
    ranges, subj_of, folds = collect_all_recordings(args.runs, args.n_folds, args.model)
    print(f"[로드] {args.dataset}: recording {len(ranges)}개 "
         f"(fold {folds} 전체 합침, model={args.model} 기준, fs={fs}Hz, seg_len={seg_len})")

    rows = []
    skipped = 0
    for (s, e), subj in zip(ranges, subj_of):
        y_raw = np.asarray(label[s:e], dtype=float)
        r = compare_recording(y_raw, fs, seg_len, do_detrend, do_bpf,
                              args.lambda_detrend, args.min_chunks)
        if r is None:
            skipped += 1
            continue
        r["participant"] = subj
        rows.append(r)

    if not rows:
        raise SystemExit(f"비교 가능한 recording이 0개입니다 (--min-chunks={args.min_chunks} "
                         f"기준). --min-chunks를 낮추거나 경로를 확인하세요.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"prv_window_agreement_{args.dataset}.csv", index=False)

    print(f"\n비교 가능한 recording: {len(df)}개 / 전체 {len(ranges)}개 "
         f"({skipped}개는 {args.min_chunks * args.eval_seconds:.0f}초 미만이라 제외)")
    print(f"평균 recording 길이: {df['duration_sec'].mean():.1f}초 "
         f"(청크 {df['n_chunks'].mean():.1f}개 평균, 범위 {df['n_chunks'].min()}~{df['n_chunks'].max()}개)")
    print(f"평균 RR 개수 — 짧은 청크 합: {df['n_rr_short'].mean():.1f}개, "
         f"긴 윈도우: {df['n_rr_long'].mean():.1f}개")

    print(f"\n{'='*70}\n짧은 청크(10초 단위) vs 긴 윈도우(recording 전체) 일치도\n{'='*70}")
    summary = {}
    for metric in ["sdnn", "rmssd", "hr"]:
        stat = bland_altman(df[f"{metric}_short"], df[f"{metric}_long"])
        unit = "bpm" if metric == "hr" else "ms"
        summary[metric] = stat
        print(f"\n--- {metric.upper()} ---")
        print(f"  n={stat['n']}  Pearson r={stat['pearson']:.3f}  "
             f"Spearman ρ={stat['spearman']:.3f}  MAE={stat['mae']:.3f}{unit}")
        print(f"  Bland-Altman: bias(짧음-긺)={stat['bias']:+.3f}{unit}  "
             f"95% LoA=[{stat['loa_lo']:+.3f}, {stat['loa_hi']:+.3f}]{unit}")

    pd.DataFrame(summary).T.to_csv(out_dir / f"prv_agreement_summary_{args.dataset}.csv")

    print(f"\n저장: {out_dir}/prv_window_agreement_{args.dataset}.csv (recording별 원자료)")
    print(f"      {out_dir}/prv_agreement_summary_{args.dataset}.csv (요약통계)")
    print("\n[판단 가이드]")
    print("  r이 높고(예: >0.7) bias가 0 근처면 -> 10초 청크 PRV가 긴 윈도우의")
    print("  편향 없는 대리지표라는 근거 (ultra-short PRV로 명시하되 신뢰 가능).")
    print("  r이 낮거나 bias가 크면 -> SDNN/RMSSD는 참고치로 격하하고 mean HR/RR")
    print("  중심으로 보고하거나, Limitations에 '10초 창의 한계'로 명시할 것.")


if __name__ == "__main__":
    main()

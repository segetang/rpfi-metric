"""
bwmd_failure_rate.py — BWMD peak-detection 실패율 집계 (R1-M10)
=====================================================================
리뷰어 지적(Major Comment 10):
  "The fallback that removes PeakDiff when peak detection fails changes the
   metric definition precisely in the lowest-quality cases and may make severe
   failures appear artificially better. Report the failure rate and use an
   explicit failure penalty or missingness rule."

compute_bwmd()가 각 10초 chunk마다 참조/예측 신호에서 피크를 몇 개 찾았는지에
따라 세 갈래로 갈린다(rpfi_eval.py 참고):
  normal  — 양쪽 다 피크 2개 이상 → RR 기반 정규화된 PeakDiff 사용 (정상 경로)
  partial — 양쪽 다 피크 1개만    → 단일 피크 시각차만 사용 (약화된 근사)
  fail    — 한쪽이라도 피크 0개   → 중립값 0.5로 대체 (원래 실패를 "적당히 나쁨"
                                     으로 채점 — 리뷰어가 우려한 지점)

이 스크립트는 fail/partial 비율을 모델·데이터셋별로 집계해 리포트한다.
RPFI 재계산(부트스트랩/집계)은 하지 않으므로 rpfi_eval.py 본실행보다 훨씬 가볍다.

사용법
------
    python bwmd_failure_rate.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" --out bwmd_failure_ubfc.csv

3개 데이터셋 다 돌리면 됨(각 --dataset/--runs/--label/--out만 교체).
"""

import argparse
import csv

import numpy as np

from rpfi_eval import FS_BY_DATASET, discover_models, load_participant_data, peak_times


def classify_chunk(y, yhat, fs):
    pt_y = peak_times(y, fs)
    pt_h = peak_times(yhat, fs)
    if len(pt_y) >= 2 and len(pt_h) >= 2:
        return "normal"
    elif len(pt_y) >= 1 and len(pt_h) >= 1:
        return "partial"
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--models", default=None, help="콤마 구분. 미지정시 자동 탐색")
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--out", default="bwmd_failure_rate.csv")
    args = ap.parse_args()

    fs = args.fs or FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    do_detrend, do_bpf = not args.no_detrend, not args.no_bpf

    label_full = np.load(args.label)
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else discover_models(args.runs))
    if not models:
        raise SystemExit(f"{args.runs} 에서 모델을 찾지 못했습니다.")

    print(f"dataset={args.dataset}  fs={fs}Hz  chunk={seg_len}samples  models={models}")
    print(f"{'model':<16s}{'n_chunk':>9s}{'normal':>9s}{'partial':>9s}{'fail':>7s}"
          f"{'fail%':>8s}{'partial%':>10s}")

    rows = []
    grand_total = {"normal": 0, "partial": 0, "fail": 0}
    for model in models:
        data = load_participant_data(args.runs, label_full, model, fs, seg_len,
                                     do_detrend, do_bpf, args.lambda_detrend)
        counts = {"normal": 0, "partial": 0, "fail": 0}
        for subj, d in data.items():
            L, P = d["label"], d["pred"]
            for c in range(len(L)):
                counts[classify_chunk(L[c], P[c], fs)] += 1
        n_total = sum(counts.values())
        if n_total == 0:
            print(f"  [건너뜀] {model}: chunk 없음")
            continue
        for k in counts:
            grand_total[k] += counts[k]

        fail_pct = 100 * counts["fail"] / n_total
        partial_pct = 100 * counts["partial"] / n_total
        print(f"{model:<16s}{n_total:>9d}{counts['normal']:>9d}{counts['partial']:>9d}"
              f"{counts['fail']:>7d}{fail_pct:>8.2f}{partial_pct:>10.2f}")
        rows.append(dict(dataset=args.dataset, model=model, n_chunks=n_total,
                         normal=counts["normal"], partial=counts["partial"],
                         fail=counts["fail"], fail_pct=round(fail_pct, 3),
                         partial_pct=round(partial_pct, 3)))

    n_all = sum(grand_total.values())
    if n_all:
        print(f"\n{'전체(모델 평균 아님, chunk 합산)':<16s}{n_all:>9d}"
              f"{grand_total['normal']:>9d}{grand_total['partial']:>9d}"
              f"{grand_total['fail']:>7d}"
              f"{100*grand_total['fail']/n_all:>8.2f}"
              f"{100*grand_total['partial']/n_all:>10.2f}")

    if rows:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()

"""
diagnose_cohface.py — COHFACE 저성능 원인 1차 진단 (모델 학습 불필요, 수 초 소요)
====================================================================
목적: "COHFACE가 원래 어려운 데이터셋이라 낮은가" vs "우리 파이프라인/추출
단계에 문제가 있어서 낮은가"를 가른다.

방법: 모델을 전혀 거치지 않고, POS 추출 결과(pred)와 true label의
recording별 raw Pearson 상관관계를 계산한다. UBFC/PURE(정상 범위로 확인된
데이터셋)와 나란히 비교한다.

해석 가이드
-----------
- COHFACE의 raw POS-label |상관|이 UBFC/PURE와 비슷한 수준(예: 0.3~0.5대)인데
  학습된 모델은 0.07~0.11밖에 안 나온다
    -> 입력엔 신호가 있는데 학습/평가 단계에서 잃고 있다는 뜻.
       windowing, z-norm, fs 정합, label 정렬 쪽을 의심할 것.
- COHFACE의 raw POS-label |상관|이 처음부터 UBFC/PURE보다 훨씬 낮고
  학습된 모델 성능(0.07~0.11)과 비슷한 수준
    -> "garbage in, garbage out". POS 추출 단계(영상 압축 등)가 병목.
       모델을 아무리 바꿔도 여기서 못 벗어난다. Discussion/Limitations에
       문헌 근거(MPEG-4 압축, McDuff et al. 2017)와 함께 기술할 것.
- flat(std≈0) 신호나 NaN이 COHFACE에서 유독 많이 나오면
    -> 얼굴 검출 실패/프레임 드롭 등 추출 파이프라인 버그일 가능성.
       해당 recording들을 개별 확인할 것.

사용법
------
    python diagnose_cohface.py \
        --ubfc-pred "../real data/UBFC_POS_prediction.npy" \
        --ubfc-label "../real data/UBFC_label.npy" \
        --ubfc-cv splits/ubfc_cv5.json \
        --cohface-pred "../real data/cohface_POS_prediction.npy" \
        --cohface-label "../real data/cohface_label.npy" \
        --cohface-cv splits/cohface_cv5.json \
        --pure-pred "../real data/PURE_POS_prediction.npy" \
        --pure-label "../real data/PURE_label.npy" \
        --pure-cv splits/pure_cv5.json
"""

import argparse
import numpy as np
from scipy.stats import pearsonr

from subject_split_v2 import load_cv, extract_by_ranges


def analyze(name, pred_path, label_path, cv_path):
    pred = np.load(pred_path)
    label = np.load(label_path)
    cv = load_cv(cv_path)

    # fold 0 하나면 전체 subject를 다 커버한다 (train+test 합치면 전체 range)
    f0 = cv["folds"][0]
    ranges = list(f0["train_ranges"]) + list(f0["test_ranges"])
    xs = extract_by_ranges(pred, ranges)
    ys = extract_by_ranges(label, ranges)

    print(f"\n{'=' * 64}\n{name}\n{'=' * 64}")
    print(f"recording 수: {len(xs)}  (fs={cv.get('seg_seconds', '?')}초 세그먼트 기준 아님, 전체 raw)")

    corrs, flat_x, flat_y, nan_x, nan_y = [], 0, 0, 0, 0
    for x, y in zip(xs, ys):
        n = min(len(x), len(y))
        if n < 10:
            continue
        xx = np.asarray(x[:n], dtype=np.float64)
        yy = np.asarray(y[:n], dtype=np.float64)
        if np.isnan(xx).any():
            nan_x += 1
        if np.isnan(yy).any():
            nan_y += 1
        if np.nanstd(xx) < 1e-8:
            flat_x += 1
            continue
        if np.nanstd(yy) < 1e-8:
            flat_y += 1
            continue
        if np.isnan(xx).any() or np.isnan(yy).any():
            continue
        r, _ = pearsonr(xx, yy)
        corrs.append(r)

    corrs = np.array(corrs)
    print(f"\nraw POS vs label Pearson (모델 없음, recording당 1개, n={len(corrs)}):")
    print(f"  mean={corrs.mean():+.4f}  median={np.median(corrs):+.4f}  "
          f"std={corrs.std():.4f}  min={corrs.min():+.4f}  max={corrs.max():+.4f}")
    print(f"  |r| median: {np.median(np.abs(corrs)):.4f}")
    print(f"  |r|<0.05 (사실상 무상관) recording: {int((np.abs(corrs) < 0.05).sum())}/{len(corrs)} "
          f"({100 * (np.abs(corrs) < 0.05).mean():.1f}%)")
    print(f"  평평한(std≈0) 신호 — POS: {flat_x}개, label: {flat_y}개")
    print(f"  NaN 포함 — POS: {nan_x}개, label: {nan_y}개")

    allx = np.concatenate([np.asarray(x, dtype=np.float64) for x in xs if len(x) > 0])
    ally = np.concatenate([np.asarray(y, dtype=np.float64) for y in ys if len(y) > 0])
    print(f"\n  신호 통계 (전체 concat, raw scale):")
    print(f"  POS   mean={allx.mean():.4f}  std={allx.std():.4f}  "
          f"min={allx.min():.4f}  max={allx.max():.4f}")
    print(f"  label mean={ally.mean():.4f}  std={ally.std():.4f}  "
          f"min={ally.min():.4f}  max={ally.max():.4f}")

    return np.abs(corrs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubfc-pred", required=True)
    ap.add_argument("--ubfc-label", required=True)
    ap.add_argument("--ubfc-cv", required=True)
    ap.add_argument("--cohface-pred", required=True)
    ap.add_argument("--cohface-label", required=True)
    ap.add_argument("--cohface-cv", required=True)
    ap.add_argument("--pure-pred", default=None)
    ap.add_argument("--pure-label", default=None)
    ap.add_argument("--pure-cv", default=None)
    args = ap.parse_args()

    results = {}
    results["UBFC"] = analyze("UBFC (비교 기준 — 정상 성능 확인됨)",
                              args.ubfc_pred, args.ubfc_label, args.ubfc_cv)
    results["COHFACE"] = analyze("COHFACE (조사 대상)",
                                 args.cohface_pred, args.cohface_label, args.cohface_cv)
    if args.pure_pred:
        results["PURE"] = analyze("PURE (비교 기준 — 정상 성능 확인됨)",
                                  args.pure_pred, args.pure_label, args.pure_cv)

    print(f"\n{'=' * 64}\n요약 — raw POS-label |Pearson| median\n{'=' * 64}")
    for k, v in results.items():
        print(f"  {k:8s}: {np.median(v):.4f}")

    print(f"\n해석:")
    print(f"  COHFACE가 UBFC/PURE와 비슷하면 -> 학습/평가 파이프라인 의심")
    print(f"  COHFACE가 UBFC/PURE보다 훨씬 낮으면 -> POS 추출 단계(입력 자체) 병목")


if __name__ == "__main__":
    main()

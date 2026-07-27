"""
export_pos_baseline.py — POS baseline을 학습 모델과 동일한 형식으로 내보낸다
==============================================================================
POS는 학습이 필요 없다 (*_POS_prediction.npy가 곧 예측이다).
다만 평가 파이프라인이 학습 모델 출력과 **동일한 방식으로** 소비할 수 있어야
공정 비교가 성립하므로, 같은 split·같은 윈도잉·같은 정규화를 적용해 내보낸다.

왜 필요한가 (리뷰어 R1-4 대응)
------------------------------
기존 논문에서 POS의 MAE는 label의 DC 오프셋과 거의 같았다:
    PURE    : POS MAE 40.1010  vs  label 평균절대값 40.1008  (0.0% 차이)
    COHFACE : POS MAE 37.2213  vs  label 평균절대값 37.0364  (0.5% 차이)
POS 출력은 평균 0인데 PURE/COHFACE label은 DC 오프셋(40.1 / 37.0)을 갖고 있어,
"오차"가 알고리즘 성능이 아니라 캘리브레이션 누락의 산물이었다.

윈도우별 z-normalize를 적용하면 이 아티팩트가 제거된다 (Pearson은 불변):
    PURE    RMSE 43.7804 -> 1.1271   Pearson +0.3649 (동일)
    UBFC    RMSE 29.1194 -> 1.3095   Pearson +0.1426 (동일)
    COHFACE RMSE 39.4890 -> 1.4145   Pearson -0.0004 (동일)

사용법
------
    # 단일 split
    python export_pos_baseline.py --pred UBFC_POS_prediction.npy \
        --split splits/ubfc_split.json --out runs/ubfc

    # K-fold CV (학습 모델과 동일한 fold 구조로 내보냄)
    python export_pos_baseline.py --pred UBFC_POS_prediction.npy \
        --cv-split splits/ubfc_cv5.json --out runs/ubfc_cv5
"""

import argparse
import json
from pathlib import Path

import numpy as np

from subject_split_v2 import (load_split, load_cv, extract_by_ranges,
                              verify_split_matches)


def znorm_windows(a):
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True)
    return (a - m) / np.maximum(s, 1e-8)


def pos_predict(x_list, seq_len):
    """POS 예측 = 입력 신호 자체. 학습 모델과 동일하게
    non-overlapping 윈도우로 자르고 윈도우별 z-normalize."""
    out = []
    for xr in x_list:
        n = (len(xr) // seq_len) * seq_len
        if n == 0:
            out.append(np.array([], dtype=np.float32)); continue
        w = znorm_windows(np.asarray(xr[:n], dtype=np.float32).reshape(-1, seq_len))
        out.append(w.reshape(-1).astype(np.float32))
    return out


def emit(pred, split, out_dir, tag=""):
    seq_len = split["seg_len"]
    te_ranges = split["test_ranges"]
    te_x = extract_by_ranges(pred, te_ranges)
    preds = pos_predict(te_x, seq_len)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"pos{tag}_test_pred.npy",
            np.array(preds, dtype=object), allow_pickle=True)
    info = dict(model="pos", fold=split.get("fold"), run_seed=None,
                n_params=0, best_val=None, epochs_run=0,
                convergence={"stop_reason": "not_trained",
                             "status": "baseline_no_training",
                             "tail_rel_gain": None, "final_lr": None},
                train_seconds=0.0,
                seq_len=int(seq_len), train_windows=0, val_windows=0,
                train_stride=None, val_subjects=[],
                train_subjects=split.get("train_subjects"),
                test_subjects=split["test_subjects"],
                test_ranges=[list(r) for r in te_ranges],
                test_subject_of_range=split["test_subject_of_range"],
                test_rec_lengths=[int(len(p)) for p in preds],
                pred_space="window_znorm",
                config=dict(note="POS baseline: 학습 없음. 입력 신호를 "
                                 "학습 모델과 동일한 윈도잉·z-normalize로 처리"))
    json.dump(info, open(out_dir / f"pos{tag}_info.json", "w"), indent=2, default=str)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="*_POS_prediction.npy")
    ap.add_argument("--split", help="단일 split JSON")
    ap.add_argument("--cv-split", help="K-fold CV JSON")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not args.split and not args.cv_split:
        ap.error("--split 또는 --cv-split 필요")

    pred = np.load(args.pred)
    ds_guess = None
    for d in ("pure", "ubfc", "cohface"):
        if d in Path(args.pred).name.lower():
            ds_guess = d
    out_dir = Path(args.out)
    infos = []

    if args.cv_split:
        cv = load_cv(args.cv_split)
        verify_split_matches(cv, pred, ds_guess, "prediction 배열")
        for f in cv["folds"]:
            fs_ = dict(f, seg_len=cv["seg_len"], seg_seconds=cv["seg_seconds"])
            infos.append(emit(pred, fs_, out_dir, tag=f"_f{f['fold']}"))
            print(f"  pos fold {f['fold']}: {sum(infos[-1]['test_rec_lengths']):,} 샘플")
    else:
        sp = load_split(args.split)
        verify_split_matches(sp, pred, ds_guess, "prediction 배열")
        infos.append(emit(pred, sp, out_dir))
        print(f"  pos: {sum(infos[-1]['test_rec_lengths']):,} 샘플")

    print(f"완료 -> {out_dir.resolve()}")


if __name__ == "__main__":
    main()

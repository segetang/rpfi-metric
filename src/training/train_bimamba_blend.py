"""
train_bimamba_blend.py — biMamba 앙상블(blend) 전용 학습 스크립트
====================================================================
원본 bimamba.py(blend 버전)의 "K개 모델을 학습시키고 최소제곱으로
블렌딩 가중치를 구한다"는 아이디어를, subject-wise 원칙을 지키면서
재현한다.

원본 대비 leakage 제거 (원본 3중 leakage 발견분)
-------------------------------------------------
| 항목 | 원본 | 이 스크립트 |
|------|------|------|
| base 모델 K-fold | `KFold(shuffle=True)`, stride=1 슬라이딩 윈도우 | **subject 단위** GroupKFold 성격 분할 |
| 정규화 | `MinMaxScaler`를 전체 배열(train+val+blend)에 fit | 윈도우별 z-norm (train_models.py와 동일, 전역 통계 누수 없음) |
| 블렌딩 가중치 학습 | 전체 배열의 마지막 20% (셔플 KFold 탓에 base 모델 학습에 이미 섞여 있었음) | base 모델 학습에 전혀 쓰이지 않는 **별도 subject 그룹** |
| 최종 평가 | 전체 배열(train+val+blend 포함, X-3과 동일한 문제) | outer test subject만, non-overlapping |
| 후처리 | Butterworth lowpass(3.0Hz) | 없음 (다른 7개 모델과 동일 조건 유지 — 공정 비교) |
| 학습 프로토콜 | 70 epoch, patience 5, CosineAnnealingWarmRestarts | 공통 하니스와 동일: epoch 상한 200, patience 15,
  SmoothL1, ReduceLROnPlateau — R1-9 "전 모델 공통 프로토콜" 주장을 깨지 않기 위해 의도적으로 통일 |
| K(base 모델 개수) | 5 | K_outer-2 = **3으로 3개 데이터셋 전부 통일** (아래 설명) |

outer fold의 5분할 그룹을 그대로 재사용한다 (v2, blend-frac 폐지)
------------------------------------------------------------------
subject_split_v2.py가 outer CV를 만들 때 이미 전체 subject를 5개의 서로
겹치지 않는 그룹으로 나눠뒀다(각 fold의 test_subjects가 그 그룹 자체다).
outer fold f의 train_subjects는 항상 "그 f를 제외한 나머지 4개 그룹의 합집합"
이라는 게 K-fold의 정의상 보장된다. 이 4개 그룹을 다시 셔플하지 않고 그대로:

  1) 4개 그룹 중 1개(고정 회전: (f+1) % K) -> blend_subjects.
     어떤 base 모델도 이 그룹을 학습에 쓰지 않는다. 블렌딩 가중치(최소제곱) 전용.
  2) 나머지 3개 그룹 -> K_inner=3개의 base 모델 각각의 "제외 그룹"(=조기종료 검증셋).
     모델 i는 이 3개 중 자기 그룹만 빼고 학습한다.

이러면 별도의 --blend-frac 하이퍼파라미터가 필요 없고, K_inner은 outer K(=5)에서
기계적으로 K-2=3으로 정해진다 — UBFC/COHFACE(그룹당 8명)와 PURE(그룹당 2명) 둘 다
"같은 규칙을 한 단계 더 적용한 것"이라 데이터셋 크기와 무관하게 절차가 통일된다.
(그룹 *크기*는 데이터셋 인원수를 따라 다르지만, 그룹 *개수*는 항상 3으로 고정.)

사용법
------
    python train_bimamba_blend.py --dataset ubfc \
        --pred "../real data/UBFC_POS_prediction.npy" \
        --label "../real data/UBFC_label.npy" \
        --cv-split splits/ubfc_cv5.json --out runs

    python train_bimamba_blend.py --dataset pure \
        --pred "../real data/PURE_POS_prediction.npy" \
        --label "../real data/PURE_label.npy" \
        --cv-split splits/pure_cv5.json --out runs

세 데이터셋 전부 커맨드가 동일하다(K_inner를 따로 지정할 필요 없음).
산출물은 기존 train_models.py와 동일한 디렉토리(runs/{dataset}_cv{K}/)에
model="bimamba_blend" 이름으로 떨어지므로, 평가 파이프라인이 그대로 소비할 수 있다.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models import build_model
from subject_split_v2 import load_cv, extract_by_ranges, verify_split_matches
from train_models import (
    znorm_windows, windows_from_recordings, make_loader, train_one,
    predict_test, eval_metrics, _stable_hash,
)


# ══════════════════════════════════════════════════════════════
# 결정론적 seed / subject 분할
# ══════════════════════════════════════════════════════════════

def _seed_for(base_seed, fold, tag):
    """train_models.py의 run_seed 패턴과 동일 원리 (md5 기반, 프로세스 무관 재현)."""
    return base_seed + 1000 * fold + (_stable_hash(tag) % 997)


def partition_subjects(all_folds, fold):
    """outer CV의 5분할 그룹을 그대로 재사용한다 (새 RNG 없음, 완전 결정론적).

    all_folds: cv["folds"] 전체 (subject_split_v2.make_cv_folds가 만든, 서로
               겹치지 않고 전체 subject를 커버하는 K개 그룹의 test_subjects).
    fold: 지금 outer test로 쓰는 fold index.

    반환: (blend_subjects, [inner_group_0, inner_group_1, ...])
          그룹 개수는 항상 len(all_folds) - 2 (outer test 1개, blend 1개를 뺀 나머지).
    """
    k_outer = len(all_folds)
    other_idx = [k for k in range(k_outer) if k != fold]
    blend_idx = other_idx[(fold + 1) % len(other_idx)]          # 고정 회전, RNG 불필요
    inner_idx = [k for k in other_idx if k != blend_idx]

    blend_subs = sorted(all_folds[blend_idx]["test_subjects"])
    groups = [sorted(all_folds[k]["test_subjects"]) for k in inner_idx]
    return blend_subs, groups


def windows_nonoverlap_znorm(x_list, y_list, seq_len):
    """non-overlapping 윈도우 + z-norm (predict_test와 동일 전처리를 label 쪽에 적용).
    반환: recording별 1D 배열 리스트."""
    yw = []
    for yr in y_list:
        n = (len(yr) // seq_len) * seq_len
        if n:
            yw.append(znorm_windows(
                np.asarray(yr[:n], dtype=np.float32).reshape(-1, seq_len)).reshape(-1))
        else:
            yw.append(np.array([], dtype=np.float32))
    return yw


# ══════════════════════════════════════════════════════════════
# inner 모델 1개 학습
# ══════════════════════════════════════════════════════════════

def train_inner_model(tag, fit_subs, val_subs, subj_x, subj_y, seq_len,
                      args, device, out_dir, fold):
    run_seed = _seed_for(args.seed, fold, f"bimamba_blend_{tag}")
    torch.manual_seed(run_seed); np.random.seed(run_seed % (2 ** 31))

    def gather(subs):
        xs, ys = [], []
        for s in subs:
            xs.extend(subj_x[s]); ys.extend(subj_y[s])
        return xs, ys

    fit_x, fit_y = gather(fit_subs)
    val_x, val_y = gather(val_subs)

    Xf, Yf = windows_from_recordings(fit_x, fit_y, seq_len, args.train_stride)
    val_stride = args.val_stride or max(1, seq_len // 4)
    Xv, Yv = windows_from_recordings(val_x, val_y, seq_len, val_stride)
    Xf, Yf = znorm_windows(Xf), znorm_windows(Yf)
    Xv, Yv = znorm_windows(Xv), znorm_windows(Yv)

    # bimamba DEFAULTS와 동일 (lr=3e-4, AdamW) + 공통 프로토콜(epoch/patience/scheduler)
    cfg = dict(lr=3e-4, epochs=args.epoch_cap, batch=32, loss=args.unify_loss, opt="adamw")
    tr_ld = make_loader(Xf, Yf, cfg["batch"], True)
    va_ld = make_loader(Xv, Yv, cfg["batch"], False)

    model = build_model("bimamba").to(device)
    n_par = sum(p.numel() for p in model.parameters())
    ck = out_dir / f"bimamba_blend_f{fold}_{tag}.pth"
    print(f"    [{tag}] params {n_par:,} | train {Xf.shape[0]} win / val {Xv.shape[0]} win "
          f"(fit={len(fit_subs)}명 / val={len(val_subs)}명)")

    hist, best, diag = train_one(model, tr_ld, va_ld, device, cfg, ck,
                                 patience=args.patience, sched_kind="plateau",
                                 smooth_beta=args.smooth_beta, verbose=not args.quiet)
    model.load_state_dict(torch.load(ck, map_location=device))

    return model, dict(tag=tag, run_seed=int(run_seed), n_params=int(n_par),
                       best_val=float(best), epochs_run=len(hist), convergence=diag,
                       fit_subjects=fit_subs, val_subjects=val_subs, ckpt=str(ck))


# ══════════════════════════════════════════════════════════════
# fold 1개 실행 (K_inner 모델 학습 + 블렌딩 + outer test 평가)
# ══════════════════════════════════════════════════════════════

def run_fold(fold_split, all_folds, pred, label, args, device, out_dir):
    fold = fold_split["fold"]
    seq_len = fold_split["seg_len"]
    train_subjects = fold_split["train_subjects"]
    test_subjects = fold_split["test_subjects"]

    tr_ranges = fold_split["train_ranges"]
    tr_subj_of = fold_split["train_subject_of_range"]
    subj_x = {s: [] for s in train_subjects}
    subj_y = {s: [] for s in train_subjects}
    for (s, e), subj in zip(tr_ranges, tr_subj_of):
        subj_x[subj].append(np.asarray(pred[s:e], dtype=np.float32))
        subj_y[subj].append(np.asarray(label[s:e], dtype=np.float32))

    blend_subs, groups = partition_subjects(all_folds, fold)
    # 무결성: outer train_subjects가 정확히 blend+groups의 합집합과 같아야 한다
    assert set(train_subjects) == set(blend_subs) | {s for g in groups for s in g}, \
        "outer train_subjects != blend_subjects ∪ inner_groups — 그룹 재사용이 깨졌습니다"
    print(f"  -- bimamba_blend fold {fold}/{fold_split.get('K', '?')} : "
          f"blend_subjects={len(blend_subs)}명, K_inner={len(groups)}, "
          f"그룹 크기={[len(g) for g in groups]}")

    models, infos = [], []
    for i, val_grp in enumerate(groups):
        fit_subs = sorted(set(train_subjects) - set(blend_subs) - set(val_grp))
        m, info = train_inner_model(f"inner{i}", fit_subs, val_grp, subj_x, subj_y,
                                    seq_len, args, device, out_dir, fold)
        models.append(m); infos.append(info)

    # ── 블렌딩 가중치: blend_subjects에서만 최소제곱으로 학습 ──
    blend_x, blend_y = [], []
    for s in blend_subs:
        blend_x.extend(subj_x[s]); blend_y.extend(subj_y[s])

    blend_preds_per_model = [predict_test(m, blend_x, seq_len, device) for m in models]
    blend_y_w = windows_nonoverlap_znorm(blend_x, blend_y, seq_len)

    flat_preds = [np.concatenate(bp) for bp in blend_preds_per_model]   # K_inner개, 각 (N,)
    P = np.stack(flat_preds, axis=1)                                   # (N, K_inner)
    y_flat = np.concatenate(blend_y_w)
    n = min(len(P), len(y_flat)); P, y_flat = P[:n], y_flat[:n]

    w, *_ = np.linalg.lstsq(P, y_flat, rcond=None)
    w = w / w.sum()
    print(f"    blend weights: {np.round(w, 4).tolist()}")

    # ── outer test subject에 가중 앙상블 적용 ──
    te_ranges = fold_split["test_ranges"]
    te_subj_of = fold_split["test_subject_of_range"]
    te_x = extract_by_ranges(pred, te_ranges)
    te_y = extract_by_ranges(label, te_ranges)

    test_preds_per_model = [predict_test(m, te_x, seq_len, device) for m in models]
    final_preds = []
    for rec_i in range(len(te_x)):
        acc = np.zeros_like(test_preds_per_model[0][rec_i])
        for wk, mp in zip(w, test_preds_per_model):
            acc = acc + wk * mp[rec_i]
        final_preds.append(acc.astype(np.float32))

    te_y_w = windows_nonoverlap_znorm(te_x, te_y, seq_len)
    metrics = eval_metrics(te_y_w, final_preds)
    print(f"    [bimamba_blend_f{fold}] test  Pearson {metrics['pearson_mean']:+.4f}  "
          f"PSNR {metrics['psnr_db']:.2f}dB  RMSE {metrics['rmse']:.4f}  "
          f"MAE {metrics['mae']:.4f}")

    np.save(out_dir / f"bimamba_blend_f{fold}_test_pred.npy",
            np.array(final_preds, dtype=object), allow_pickle=True)

    info = dict(
        model="bimamba_blend", fold=fold,
        k_inner=len(groups),
        blend_group_source="outer_cv_fold_reuse",  # blend/inner 그룹이 outer 5분할을 재사용했다는 표시
        blend_subjects=blend_subs, inner_groups=groups,
        blend_weights=[float(x) for x in w],
        inner_models=infos,
        train_subjects=train_subjects, test_subjects=test_subjects,
        test_ranges=[list(r) for r in te_ranges],
        test_subject_of_range=te_subj_of,
        test_rec_lengths=[int(len(p)) for p in final_preds],
        pred_space="window_znorm",
        test_metrics=metrics,
        config=dict(lr=3e-4, opt="adamw", loss=args.unify_loss,
                   epoch_cap=args.epoch_cap, patience=args.patience,
                   smooth_beta=args.smooth_beta, train_stride=args.train_stride,
                   scheduler="plateau"),
    )
    json.dump(info, open(out_dir / f"bimamba_blend_f{fold}_info.json", "w"),
              indent=2, default=str)
    return info


# ══════════════════════════════════════════════════════════════
# 실행 단위
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--cv-split", required=True, help="K-fold CV JSON (subject_split_v2.py --cv K)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--folds", default=None, help="특정 fold만 실행 (콤마 구분). 예: --folds 0,1")
    ap.add_argument("--train-stride", type=int, default=10)
    ap.add_argument("--val-stride", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--epoch-cap", type=int, default=200)
    ap.add_argument("--unify-loss", choices=["mse", "smoothl1"], default="smoothl1")
    ap.add_argument("--smooth-beta", type=float, default=1.0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = np.load(args.pred); label = np.load(args.label)

    cv = load_cv(args.cv_split)
    verify_split_matches(cv, pred, args.dataset, "prediction 배열")
    verify_split_matches(cv, label, args.dataset, "label 배열")

    out_dir = Path(args.out) / f"{args.dataset}_cv{cv['K']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    sel_folds = None
    if args.folds:
        sel_folds = {int(x) for x in args.folds.split(",") if x.strip()}

    print(f"dataset={args.dataset}  bimamba-blend  {cv['K']}-fold subject CV  "
          f"K_inner={cv['K'] - 2} (outer 그룹 재사용: 1개 blend + {cv['K'] - 2}개 inner)  "
          f"device={device}")

    summary = []
    for f in cv["folds"]:
        if sel_folds is not None and f["fold"] not in sel_folds:
            continue
        done = out_dir / f"bimamba_blend_f{f['fold']}_info.json"
        if args.skip_existing and done.exists():
            print(f"  -- fold {f['fold']}: 이미 존재, 건너뜀")
            summary.append(json.load(open(done)))
            continue
        fold_split = dict(f, seg_len=cv["seg_len"], seg_seconds=cv["seg_seconds"], K=cv["K"])
        t0 = time.time()
        info = run_fold(fold_split, cv["folds"], pred, label, args, device, out_dir)
        print(f"    ({time.time() - t0:.1f}s)")
        summary.append(info)

    pr = [r["test_metrics"].get("pearson_mean", np.nan) for r in summary]
    ps = [r["test_metrics"].get("psnr_db", np.nan) for r in summary]
    print(f"\n=== bimamba_blend fold 집계 ===  n_fold={len(summary)}  "
          f"Pearson {np.nanmean(pr):+.4f}±{np.nanstd(pr):.4f}  PSNR {np.nanmean(ps):.2f}dB")

    bad = [r["fold"] for r in summary
           if any(im["convergence"]["status"] == "capped_still_improving"
                  for im in r.get("inner_models", []))]
    if bad:
        print(f"  ⚠️ 학습 부족 의심 fold: {bad} (inner 모델 중 일부가 capped_still_improving)")
    else:
        print("  ✓ 모든 inner 모델이 수렴 조건으로 종료됨 (학습 부족 없음)")

    # summary.json 병합 (train_models.py의 다른 모델 결과와 같은 파일에 누적)
    sp = out_dir / "summary.json"
    merged = {}
    if sp.exists():
        for r in json.load(open(sp)):
            merged[(r["model"], r.get("fold"))] = r
    for r in summary:
        merged[(r["model"], r.get("fold"))] = r
    json.dump(list(merged.values()), open(sp, "w"), indent=2, default=str)
    print(f"완료 -> {out_dir.resolve()}  (summary 누적 {len(merged)}건)")


if __name__ == "__main__":
    main()

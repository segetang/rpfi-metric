"""
train_models.py — 전 모델 공통 학습/예측 하니스
================================================
기존 개별 스크립트(RNN.py, LSTM.py, bilstm_test.py, physdiff.py,
mamba.py, bimamba(plain).py)를 대체한다. 모든 모델이 **동일한 split·전처리·
평가 절차**를 쓰도록 강제하여 리뷰어가 요구한 공정 비교를 보장한다.

기존 스크립트 대비 핵심 변경 (subject_split_v2 도입에 따른)
-----------------------------------------------------------
| 항목 | 기존 | 변경 |
|------|------|------|
| split | `int(len(data)*0.8)` 시간축 분할 | mapping CSV 기반 **participant 단위** |
| physdiff | `random_split`(윈도우 랜덤) | 제거 |
| bimamba | `KFold(shuffle=True)` | 제거 (plain만) |
| 윈도우 | 전체 배열에 stride=1 슬라이딩 | **recording 경계 내부**에서만 생성 |
| 학습/평가 stride | 동일 | 학습은 조밀(augmentation), **평가는 non-overlapping** |
| 예측 저장 | train+test 전 구간 | **test subject만** |
| 정규화 | MinMaxScaler(전역) 또는 없음 | 윈도우별 z-norm (전역 통계 누수 없음) |
| 검증셋 | test를 val로 겸용 | train subject 중 일부를 **subject 단위**로 분리 |
| seed | 미고정 | 42 고정 |

사용법
------
    # 1) split 생성 (데이터셋별 1회)
    python subject_split_v2.py --dataset all --mapping-dir . --out-dir splits

    # 2) 모델 학습 (하나씩 또는 --model all)
    python train_models.py --model bilstm --dataset ubfc \
        --pred UBFC_POS_prediction.npy --label UBFC_label.npy \
        --split splits/ubfc_split.json --out runs/

    # 3) PURE는 subject가 10명뿐이라 단일 split 불가 -> CV 사용
    python train_models.py --model bilstm --dataset pure ... --cv 5
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from models import build_model, DEFAULTS, MODEL_REGISTRY
from subject_split_v2 import (load_split, load_cv, extract_by_ranges,
                              verify_split_matches)


# ══════════════════════════════════════════════════════════════
# 전처리 — 평가 파이프라인과 통일
# ══════════════════════════════════════════════════════════════

def znorm_windows(a):
    """윈도우별 z-normalize. 전역 통계를 쓰지 않으므로 split 간 누수 없음.

    ※ 기존 스크립트들은 MinMaxScaler를 **전체 배열**에 fit 했는데,
      이는 test 구간의 min/max가 train 정규화에 반영되는 누수였다.
    """
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True)
    return (a - m) / np.maximum(s, 1e-8)


def windows_from_recordings(x_list, y_list, seq_len, stride):
    """recording 경계를 넘지 않는 윈도우 생성."""
    xs, ys = [], []
    for xr, yr in zip(x_list, y_list):
        n = min(len(xr), len(yr))
        for i in range(0, n - seq_len + 1, stride):
            xs.append(xr[i:i + seq_len]); ys.append(yr[i:i + seq_len])
    if not xs:
        raise ValueError("윈도우가 생성되지 않았습니다. seq_len/stride를 확인하세요.")
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════════

def make_loader(X, Y, batch, shuffle):
    ds = TensorDataset(torch.from_numpy(X).float().unsqueeze(-1),
                       torch.from_numpy(Y).float())
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def train_one(model, tr_ld, va_ld, device, cfg, ckpt, patience=15,
              sched_kind="plateau", smooth_beta=1.0, min_lr=1e-6, verbose=True):
    """공통 학습 루프.

    epoch 상한 문제 대응:
      - sched_kind="plateau": ReduceLROnPlateau. 검증손실이 정체하면 LR을 낮추고,
        LR이 min_lr 아래로 내려가면 '더 학습해도 얻을 게 없는 상태'로 보고 종료한다.
        epoch 예산이 아니라 **수렴 여부**가 종료를 결정하므로 상한 의존성이 사라진다.
      - 종료 후 상한 도달 여부와 '마지막 구간에서 아직 개선 중이었는지'를 함께 진단한다.
    """
    crit = {"mse": nn.MSELoss(),
            "smoothl1": nn.SmoothL1Loss(beta=smooth_beta)}[cfg["loss"]]
    if cfg["opt"] == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    if sched_kind == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=max(3, patience // 3))
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    best, wait, hist = float("inf"), 0, []
    stop_reason = "epoch_cap"
    for ep in range(1, cfg["epochs"] + 1):
        model.train(); tot = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * xb.size(0)
        tr = tot / len(tr_ld.dataset)

        model.eval(); tot = 0.0
        with torch.no_grad():
            for xb, yb in va_ld:
                xb, yb = xb.to(device), yb.to(device)
                tot += crit(model(xb), yb).item() * xb.size(0)
        va = tot / len(va_ld.dataset)
        hist.append((tr, va))

        if sched_kind == "plateau":
            sched.step(va)
        else:
            sched.step()
        cur_lr = opt.param_groups[0]["lr"]

        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"    ep {ep:3d}/{cfg['epochs']}  train {tr:.5f}  val {va:.5f}  lr {cur_lr:.2e}")

        if va < best - 1e-7:
            best, wait = va, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= patience:
                stop_reason = "early_stop"; break
        if cur_lr < min_lr:
            stop_reason = "lr_exhausted"; break

    # ── 수렴 진단: 상한에 걸렸어도 '이미 평탄해졌는지' 구분 ──
    vals = np.array([v for _, v in hist], dtype=float)
    tail = max(5, len(vals) // 5)                      # 마지막 20% 구간
    if len(vals) >= 2 * tail:
        prev_best = float(vals[:-tail].min())
        rel_gain = (prev_best - float(vals[-tail:].min())) / max(abs(prev_best), 1e-12)
    else:
        rel_gain = float("nan")

    capped = (stop_reason == "epoch_cap")
    if not capped:
        status = f"converged({stop_reason})"
    elif np.isnan(rel_gain):
        status = "capped_still_improving"        # 판정 불가 -> 보수적으로 학습부족 처리
    elif rel_gain < 0.005:
        status = "capped_but_plateaued"          # 상한에 걸렸으나 사실상 수렴
    else:
        status = "capped_still_improving"        # 진짜 학습 부족

    return hist, best, dict(stop_reason=stop_reason, status=status,
                            tail_rel_gain=None if np.isnan(rel_gain) else round(rel_gain, 5),
                            final_lr=float(opt.param_groups[0]["lr"]))


def eval_metrics(y_list, p_list):
    """test 예측 즉시 확인용 지표. RPFI 본 평가는 별도 파이프라인에서 수행한다.
    (여기서는 학습이 정상적으로 되었는지 빠르게 판별하는 용도)"""
    from scipy.stats import pearsonr
    rs, ys, ps = [], [], []
    for y, p in zip(y_list, p_list):
        n = min(len(y), len(p))
        if n < 10:
            continue
        y, p = np.asarray(y[:n], float), np.asarray(p[:n], float)
        if y.std() > 1e-8 and p.std() > 1e-8:
            rs.append(pearsonr(y, p)[0])
        ys.append(y); ps.append(p)
    if not ys:
        return dict(pearson_mean=float("nan"), rmse=float("nan"),
                    mae=float("nan"), psnr_db=float("nan"), n_recordings=0)
    Y, P = np.concatenate(ys), np.concatenate(ps)
    mse = float(np.mean((Y - P) ** 2))
    peak = float(np.max(np.abs(Y)))
    return dict(pearson_mean=float(np.nanmean(rs)) if rs else float("nan"),
                pearson_sd=float(np.nanstd(rs)) if rs else float("nan"),
                rmse=float(np.sqrt(mse)), mae=float(np.mean(np.abs(Y - P))),
                psnr_db=float(10 * np.log10(peak ** 2 / mse)) if mse > 0 else float("inf"),
                n_recordings=len(ys))


@torch.no_grad()
def predict_test(model, te_x_list, seq_len, device, batch=64):
    """test recording에 대해서만 non-overlapping 예측.
    반환: recording별 1D 배열 리스트 (RPFI 파이프라인의 subject 구조에 대응)."""
    model.eval(); out = []
    for xr in te_x_list:
        n = (len(xr) // seq_len) * seq_len
        if n == 0:
            out.append(np.array([], dtype=np.float32)); continue
        w = znorm_windows(np.asarray(xr[:n], dtype=np.float32).reshape(-1, seq_len))
        t = torch.from_numpy(w).float().unsqueeze(-1).to(device)
        preds = [model(t[i:i + batch]).cpu().numpy() for i in range(0, len(t), batch)]
        out.append(np.concatenate(preds).reshape(-1).astype(np.float32))
    return out


# ══════════════════════════════════════════════════════════════
# 실행 단위
# ══════════════════════════════════════════════════════════════

def _stable_hash(s: str) -> int:
    """파이썬 내장 hash()는 PYTHONHASHSEED에 따라 프로세스마다 랜덤화되어
    같은 문자열도 실행할 때마다 다른 값을 반환한다 (hash DoS 방지용 보안 기능).
    seed 계산처럼 실행 간 재현성이 필요한 곳에는 절대 쓰면 안 된다.
    md5는 프로세스와 무관하게 항상 동일한 값을 준다."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def run(model_name, pred, label, split, args, device, out_dir, tag=""):
    # fold/모델별로 seed를 재설정해야 개별 실행이 독립적으로 재현 가능하다.
    # (main에서 한 번만 seed를 잡으면 실행 순서가 결과를 바꾼다)
    run_seed = args.seed + 1000 * split.get("fold", 0) + (_stable_hash(model_name) % 997)
    torch.manual_seed(run_seed); np.random.seed(run_seed % (2**31))
    seq_len = split["seg_len"]
    tr_ranges, te_ranges = split["train_ranges"], split["test_ranges"]
    tr_subj_of = split["train_subject_of_range"]

    tr_x = extract_by_ranges(pred, tr_ranges);  tr_y = extract_by_ranges(label, tr_ranges)
    te_x = extract_by_ranges(pred, te_ranges);  te_y = extract_by_ranges(label, te_ranges)

    # ── train subject 중 일부를 검증용으로 분리 (subject 단위, test 미사용) ──
    subs = sorted(set(tr_subj_of))
    rng = np.random.default_rng(run_seed)
    perm = np.array(subs, dtype=object); rng.shuffle(perm)
    n_val = max(1, int(round(len(subs) * args.val_frac)))
    val_subs = set(perm[:n_val].tolist())

    fit_i = [i for i, s in enumerate(tr_subj_of) if s not in val_subs]
    val_i = [i for i, s in enumerate(tr_subj_of) if s in val_subs]

    Xf, Yf = windows_from_recordings([tr_x[i] for i in fit_i], [tr_y[i] for i in fit_i],
                                     seq_len, args.train_stride)
    val_stride = args.val_stride or max(1, seq_len // 4)
    Xv, Yv = windows_from_recordings([tr_x[i] for i in val_i], [tr_y[i] for i in val_i],
                                     seq_len, val_stride)
    Xf, Yf = znorm_windows(Xf), znorm_windows(Yf)
    Xv, Yv = znorm_windows(Xv), znorm_windows(Yv)

    cfg = dict(DEFAULTS[model_name])
    if args.epochs:      cfg["epochs"] = args.epochs
    if args.batch:       cfg["batch"] = args.batch
    if args.unify_loss:  cfg["loss"] = args.unify_loss
    if args.epoch_cap:   cfg["epochs"] = args.epoch_cap

    tr_ld = make_loader(Xf, Yf, cfg["batch"], True)
    va_ld = make_loader(Xv, Yv, cfg["batch"], False)

    kw = {}
    if model_name == "transformer":
        kw["patch_size"] = args.patch_size
    model = build_model(model_name, **kw).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    print(f"  [{model_name}] params {n_par:,} | train {Xf.shape[0]} win / "
          f"val {Xv.shape[0]} win / test {len(te_x)} rec")

    ck = out_dir / f"{model_name}{tag}.pth"
    t0 = time.time()
    hist, best, diag = train_one(model, tr_ld, va_ld, device, cfg, ck,
                                 patience=args.patience, sched_kind=args.scheduler,
                                 smooth_beta=args.smooth_beta,
                                 verbose=not args.quiet)
    elapsed = time.time() - t0

    g = diag["tail_rel_gain"]
    gs = "측정불가(epoch 부족)" if g is None else f"{g:.3%}"
    if diag["status"] == "capped_still_improving":
        print(f"    ⚠️ [{model_name}{tag}] epoch 상한({cfg['epochs']}) 도달, "
              f"마지막 구간 개선률 {gs} — 학습 부족. --epoch-cap 상향 필요")
    elif diag["status"] == "capped_but_plateaued":
        print(f"    [{model_name}{tag}] 상한 도달했으나 평탄화됨 "
              f"(마지막 구간 개선률 {gs}) — 수렴으로 간주")

    model.load_state_dict(torch.load(ck, map_location=device))
    preds = predict_test(model, te_x, seq_len, device)

    # ── test 메트릭 (예측과 동일한 윈도잉·z-norm을 label에 적용) ──
    te_y_w = []
    for yr in te_y:
        n = (len(yr) // seq_len) * seq_len
        if n:
            te_y_w.append(znorm_windows(
                np.asarray(yr[:n], dtype=np.float32).reshape(-1, seq_len)).reshape(-1))
        else:
            te_y_w.append(np.array([], dtype=np.float32))
    metrics = eval_metrics(te_y_w, preds)
    print(f"    [{model_name}{tag}] test  Pearson {metrics['pearson_mean']:+.4f}  "
          f"PSNR {metrics['psnr_db']:.2f}dB  RMSE {metrics['rmse']:.4f}  "
          f"MAE {metrics['mae']:.4f}")

    np.save(out_dir / f"{model_name}{tag}_test_pred.npy",
            np.array(preds, dtype=object), allow_pickle=True)
    info = dict(model=model_name, fold=split.get("fold"),
                run_seed=int(run_seed),
                n_params=int(n_par), best_val=float(best),
                epochs_run=len(hist), convergence=diag,
                train_seconds=round(elapsed, 1),
                seq_len=int(seq_len), train_windows=int(Xf.shape[0]),
                val_windows=int(Xv.shape[0]),
                train_stride=int(args.train_stride), val_stride=int(val_stride),
                val_subjects=sorted(val_subs),
                train_subjects=split.get("train_subjects"),
                test_subjects=split["test_subjects"],
                # ── 평가 파이프라인이 label을 동일하게 슬라이싱하기 위한 정보 ──
                test_ranges=[list(r) for r in te_ranges],
                test_subject_of_range=split["test_subject_of_range"],
                test_rec_lengths=[int(len(p)) for p in preds],
                pred_space="window_znorm",   # 예측은 윈도우별 z-space (원 스케일 아님)
                test_metrics=metrics,
                config=cfg)
    json.dump(info, open(out_dir / f"{model_name}{tag}_info.json", "w"),
              indent=2, default=str)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    help="모델명(콤마 구분 가능) 또는 all. "
                         "예: --model rnn,lstm,bilstm / 선택: " + ", ".join(MODEL_REGISTRY))
    ap.add_argument("--folds", default=None,
                    help="CV에서 특정 fold만 실행 (콤마 구분). 예: --folds 0,1,2")
    ap.add_argument("--skip-existing", action="store_true",
                    help="이미 결과(info.json)가 있으면 건너뜀. 나눠서 실행할 때 사용")
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--pred", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--split", help="단일 split JSON (subject_split_v2.py)")
    ap.add_argument("--cv-split", help="K-fold CV JSON (subject_split_v2.py --cv K)")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--train-stride", type=int, default=10,
                    help="학습 윈도우 stride. train subject 내부 겹침은 leakage가 아님")
    ap.add_argument("--val-frac", type=float, default=0.20,
                    help="train subject 중 검증용 비율 (subject 단위)")
    ap.add_argument("--val-stride", type=int, default=None,
                    help="검증 윈도우 stride. 기본 seq_len//4. "
                         "val subject는 train/test와 분리되어 있으므로 "
                         "겹침 윈도우를 써도 leakage가 아니며, 조기종료 판단이 안정된다")
    ap.add_argument("--patch-size", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None, help="지정 시 DEFAULTS 덮어씀")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--unify-loss", choices=["mse", "smoothl1"], default="smoothl1",
                    help="전 모델 loss 통일 (기본 smoothl1)")
    ap.add_argument("--smooth-beta", type=float, default=1.0,
                    help="SmoothL1 전환점. z-norm 데이터에서 잔차 범위가 [0,2]이므로 "
                         "beta=1.0이면 상당 부분이 선형 구간. 낮출수록 MSE에 가까움. "
                         "논문에 반드시 명시할 것")
    ap.add_argument("--epoch-cap", type=int, default=200,
                    help="전 모델 공통 epoch 상한 (기본 200)")
    ap.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau",
                    help="plateau=ReduceLROnPlateau(권장, 수렴이 종료를 결정) / cosine")
    args = ap.parse_args()
    if not args.split and not args.cv_split:
        ap.error("--split 또는 --cv-split 중 하나는 필요합니다")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pred = np.load(args.pred); label = np.load(args.label)
    if args.model == "all":
        names = list(MODEL_REGISTRY)
    else:
        names = [m.strip() for m in args.model.split(",") if m.strip()]
        unknown = [m for m in names if m not in MODEL_REGISTRY]
        if unknown:
            raise SystemExit(f"알 수 없는 모델: {unknown} / 사용 가능: {list(MODEL_REGISTRY)}")
    sel_folds = None
    if args.folds:
        sel_folds = {int(x) for x in args.folds.split(",") if x.strip()}
    summary = []

    if args.cv_split:
        cv = load_cv(args.cv_split)
        verify_split_matches(cv, pred, args.dataset, "prediction 배열")
        verify_split_matches(cv, label, args.dataset, "label 배열")
        out_dir = Path(args.out) / f"{args.dataset}_cv{cv['K']}"
        out_dir.mkdir(parents=True, exist_ok=True)
        loso = " (LOSO)" if cv["K"] == cv["n_subjects_total"] else ""
        print(f"dataset={args.dataset}  {cv['K']}-fold subject CV{loso}  "
              f"seg_len={cv['seg_len']} ({cv['seg_seconds']}s)  "
              f"subjects={cv['n_subjects_total']}  device={device}")
        for nm in names:
            for f in cv["folds"]:
                if sel_folds is not None and f["fold"] not in sel_folds:
                    continue
                done = out_dir / f"{nm}_f{f['fold']}_info.json"
                if args.skip_existing and done.exists():
                    print(f"  -- {nm} fold {f['fold']}: 이미 존재, 건너뜀")
                    summary.append(json.load(open(done)))
                    continue
                fold_split = dict(f, seg_len=cv["seg_len"],
                                  seg_seconds=cv["seg_seconds"])
                try:
                    print(f"  -- {nm} fold {f['fold']}/{cv['K']-1} "
                          f"(test subjects: {len(f['test_subjects'])}명)")
                    summary.append(run(nm, pred, label, fold_split, args, device,
                                       out_dir, tag=f"_f{f['fold']}"))
                except ImportError as e:
                    print(f"  [{nm}] 건너뜀: {e}"); break
                except Exception as e:
                    print(f"  [{nm} f{f['fold']}] 실패: {type(e).__name__}: {e}")
        # fold 집계
        agg = {}
        for r in summary:
            agg.setdefault(r["model"], []).append(r["best_val"])
        print("\n=== fold 집계 ===")
        magg = {}
        for r in summary:
            magg.setdefault(r["model"], []).append(r.get("test_metrics", {}))
        for m, vs in agg.items():
            ms = magg.get(m, [])
            pr = [x.get("pearson_mean", np.nan) for x in ms if x]
            ps = [x.get("psnr_db", np.nan) for x in ms if x]
            print(f"  {m:12s} n_fold={len(vs)}  val {np.mean(vs):.5f}±{np.std(vs):.5f}  "
                  f"Pearson {np.nanmean(pr):+.4f}±{np.nanstd(pr):.4f}  "
                  f"PSNR {np.nanmean(ps):.2f}dB")
        bad = [f"{r['model']}_f{r['fold']}" for r in summary
               if r["convergence"]["status"] == "capped_still_improving"]
        if bad:
            print(f"\n  ⚠️ 학습 부족 의심 {len(bad)}건: {bad}")
        else:
            print("\n  ✓ 모든 실행이 수렴 조건으로 종료됨 (학습 부족 없음)")
    else:
        split = load_split(args.split)
        verify_split_matches(split, pred, args.dataset, "prediction 배열")
        verify_split_matches(split, label, args.dataset, "label 배열")
        out_dir = Path(args.out) / args.dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"dataset={args.dataset}  seg_len={split['seg_len']} "
              f"({split['seg_seconds']}s)  subjects {split['n_subjects_train']}tr/"
              f"{split['n_subjects_test']}te  device={device}")
        for nm in names:
            done = out_dir / f"{nm}_info.json"
            if args.skip_existing and done.exists():
                print(f"  [{nm}] 이미 존재, 건너뜀")
                summary.append(json.load(open(done))); continue
            try:
                summary.append(run(nm, pred, label, split, args, device, out_dir))
            except ImportError as e:
                print(f"  [{nm}] 건너뜀: {e}")
            except Exception as e:
                print(f"  [{nm}] 실패: {type(e).__name__}: {e}")

    # 부분 실행을 반복해도 전체 요약이 유지되도록 기존 결과와 병합
    sp = out_dir / "summary.json"
    merged = {}
    if sp.exists():
        for r in json.load(open(sp)):
            merged[(r["model"], r.get("fold"))] = r
    for r in summary:
        merged[(r["model"], r.get("fold"))] = r
    json.dump(list(merged.values()), open(sp, "w"), indent=2, default=str)
    print(f"\n완료 -> {out_dir.resolve()}  (summary 누적 {len(merged)}건)")


if __name__ == "__main__":
    main()

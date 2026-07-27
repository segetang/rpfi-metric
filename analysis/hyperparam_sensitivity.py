"""
hyperparam_sensitivity.py — 나머지 하이퍼파라미터 민감도 (M11 대응, REAL 파이프라인판)
=============================================================================
rpfi_eval.py 실제 소스를 확인하고 전면 재작성. 더 이상 폴백/추측 없음 —
모든 저수준 함수(dtw_distance, peak_times, median_rr, spectral_snr, sigmoid,
compute_wcr, compute_edd, load_participant_data, normalize_component, aggregate)
를 rpfi_eval.py에서 **직접 import**해서 쓴다.

BWMD/SRE만 하이퍼파라미터가 모듈 상수(BWMD_LAMBDA_P/A, SRE_SNR_LOW/HIGH 등)로
박혀있어 함수 인자로 못 넘기므로, compute_bwmd()/compute_sre()의 조합 공식을
그대로 복제하되 그 값들을 인자로 노출한 래퍼(bwmd_param/sre_param)만 새로
작성했다. 두 래퍼 모두 rpfi_eval의 실제 저수준 함수(dtw_distance, peak_times,
median_rr, spectral_snr, sigmoid, _trapz)를 그대로 호출하므로, baseline 값
(λp=1.0, λa=0.5, dtw_band=0.1, Sc=3, Ss=15)에서 실행하면 rpfi_eval.compute_bwmd
/compute_sre와 완전히 동일한 결과가 나온다 — 실행 시 self-check로 자동 검증.

방법론 (§4-f 가중치 민감도와 동일 스타일 — weight_sensitivity()를 그대로 참고)
--------------------------------------------------------------------------
  1) baseline 하이퍼파라미터로 모델별 "컴포넌트 평균"을 구하고 1회 정규화+집계
     (weight_sensitivity가 per_model_comp_means를 쓰는 것과 동일한 순서:
      "참여자별 RPFI를 낸 뒤 평균"이 아니라 "원값을 평균한 뒤 1회 정규화+집계")
  2) 각 파라미터를 그리드로 OAT(one-at-a-time) 스윕, 그때마다 전체 재계산
  3) --joint-random N: 전 파라미터를 동시에 균등분포에서 N회 무작위 추출
  4) baseline과의 Spearman ρ / Kendall τ, 1위·꼴찌 모델 유지 여부를 기록

성능 참고: DTW가 순수 python 이중루프라 전체 청크를 다 돌리면 느리다.
--n-participants/--n-chunks-per-participant로 서브샘플링하되, **전처리
(detrend+BPF, load_participant_data 결과)는 1회만 캐싱**하고 하이퍼파라미터
스윕 때는 캐시된 청크에 대해 BWMD/SRE/WCR/EDD만 재계산한다 — 즉 파라미터가
바뀌어도 detrend_tarvainen(O(N^3) 행렬역산)은 절대 다시 안 돈다.

사용법
------
    python hyperparam_sensitivity.py \
        --dataset ubfc --runs runs/ubfc_cv5 --label "../real data/UBFC_label.npy" \
        --anchors anchors_ubfc.json \
        --param dtw_band_frac

    # 전 파라미터 동시 무작위 스윕 (가중치 민감도와 나란히 보고할 형태)
    python hyperparam_sensitivity.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" --anchors anchors_ubfc.json \
        --joint-random 300

★ --anchors 를 반드시 derive_anchors.py가 뽑은 실제 anchor JSON으로 지정할 것.
  rpfi_eval.py 안의 DEFAULT_ANCHORS는 스크립트 자체 주석(862~867행)에 적혀있듯
  SRE 부호수정 이전 값(스케일이 4.4배 다름)이라 그대로 쓰면 anchor가 안 맞는다.
  --anchors 를 안 주면 이전 세션에서 derive_anchors.py로 확정했던 값
  (BWMD[0.000621,0.902266], SRE[0.003767,1.485384])을 기본값으로 쓰지만,
  실제 파일로 다시 확인해서 --anchors로 넘기는 걸 강력히 권장.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

# ══════════════════════════════════════════════════════════════
# rpfi_eval.py 직접 import (같은 폴더 또는 PYTHONPATH에 있어야 함)
# ══════════════════════════════════════════════════════════════
try:
    import rpfi_eval as R
except ImportError as e:
    sys.exit(f"[오류] rpfi_eval.py를 import할 수 없습니다 ({e}).\n"
             f"       이 스크립트를 rpfi_eval.py와 같은 폴더에서 실행하거나,\n"
             f"       PYTHONPATH에 그 폴더를 추가하세요.")

# derive_anchors.py로 확정했던 최종 anchor (SRE 부호수정 반영 후). --anchors로 덮어쓰길 권장.
FALLBACK_ANCHORS = {"BWMD": (0.000621, 0.902266), "SRE": (0.003767, 1.485384),
                    "WCR": (0.0, 1.0), "EDD": (0.0, 1.0)}


# ══════════════════════════════════════════════════════════════
# BWMD/SRE 파라미터화 래퍼 — rpfi_eval의 실제 저수준 함수를 그대로 재사용,
# 조합 공식만 compute_bwmd()/compute_sre()에서 그대로 복제 (모듈 상수 대신 인자로)
# ══════════════════════════════════════════════════════════════

def bwmd_param(y, yhat, fs, lambda_p, lambda_a, dtw_band_frac):
    """rpfi_eval.compute_bwmd()와 동일 공식, λp/λa/DTW band만 인자로 노출."""
    a = y / (np.max(np.abs(y)) + 1e-8)
    b = yhat / (np.max(np.abs(yhat)) + 1e-8)
    dtw_norm = R.dtw_distance(a, b, sakoe_chiba_ratio=dtw_band_frac)

    pt_y, pt_h = R.peak_times(y, fs), R.peak_times(yhat, fs)
    if len(pt_y) >= 2 and len(pt_h) >= 2:
        k = min(len(pt_y), len(pt_h))
        diff_mean = float(np.mean(np.abs(pt_y[:k] - pt_h[:k])))
        rr = R.median_rr(pt_y)
        rr_norm = rr if (rr and rr > 1e-6) else (len(y) / float(fs))
        peak_norm = diff_mean / rr_norm
    elif len(pt_y) >= 1 and len(pt_h) >= 1:
        peak_norm = abs(pt_y[0] - pt_h[0]) / (len(y) / float(fs))
    else:
        peak_norm = 0.5
    peak_norm = float(np.clip(peak_norm, 0.0, 1.0))

    abs_y, abs_h = np.abs(y), np.abs(yhat)
    inter = float(R._trapz(np.minimum(abs_y, abs_h)))
    union = float(R._trapz(np.maximum(abs_y, abs_h)))
    area_overlap = inter / (union + 1e-8)

    return float((dtw_norm + lambda_p * peak_norm) / (1.0 + lambda_a * area_overlap))


def sre_param(y, yhat, fs, Sc, Ss, w_low=None, w_high=None, g_min=None, g_max=None):
    """rpfi_eval.compute_sre()와 동일 공식, Sc/Ss만 인자로 노출.
    w_low/w_high/g_min/g_max는 M11 범위 밖이라 기본은 rpfi_eval 상수 그대로 사용."""
    w_low = R.SRE_W_LOW if w_low is None else w_low
    w_high = R.SRE_W_HIGH if w_high is None else w_high
    g_min = R.SRE_G_MIN if g_min is None else g_min
    g_max = R.SRE_G_MAX if g_max is None else g_max

    e = np.ravel(yhat) - np.ravel(y)
    rmse = float(np.sqrt(np.mean(e ** 2))) if e.size else 0.0
    snr_db = R.spectral_snr(y, fs)

    r_low = max(0.0, (Sc - snr_db) / Sc)
    r_high = max(0.0, (snr_db - Ss) / Ss)
    cri = R.sigmoid(Sc - snr_db)
    si = R.sigmoid(snr_db - Ss)
    g = 1.0 + w_low * cri * r_low - w_high * si * r_high
    g = float(np.clip(g, g_min, g_max))
    return float(rmse * g)


def _self_check(fs=30):
    """baseline 파라미터에서 파라미터화 래퍼가 rpfi_eval의 실제 함수와
    100% 동일한 값을 내는지 실행 시점에 검증. 다르면 즉시 중단(조용히 틀린
    숫자를 내는 것보다 낫다)."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 10, 1 / fs)
    y = np.sin(2 * np.pi * 1.2 * t) + 0.05 * rng.standard_normal(len(t))
    yhat = np.sin(2 * np.pi * 1.2 * t + 0.05) + 0.08 * rng.standard_normal(len(t))

    b_real, *_ = R.compute_bwmd(y, yhat, fs)
    b_mine = bwmd_param(y, yhat, fs, R.BWMD_LAMBDA_P, R.BWMD_LAMBDA_A, 0.1)
    s_real, *_ = R.compute_sre(y, yhat, fs)
    s_mine = sre_param(y, yhat, fs, R.SRE_SNR_LOW, R.SRE_SNR_HIGH)

    if not (np.isclose(b_real, b_mine) and np.isclose(s_real, s_mine)):
        sys.exit(f"[self-check 실패] 파라미터화 래퍼가 rpfi_eval 원본과 다릅니다!\n"
                 f"  BWMD: real={b_real:.8f} mine={b_mine:.8f}\n"
                 f"  SRE : real={s_real:.8f} mine={s_mine:.8f}\n"
                 f"  rpfi_eval.py가 이 스크립트를 짠 이후 수정됐을 수 있습니다. "
                 f"위 래퍼를 최신 compute_bwmd/compute_sre와 다시 대조하세요.")
    print(f"[self-check 통과] baseline에서 BWMD={b_real:.6f}, SRE={s_real:.6f} "
         f"(파라미터화 래퍼와 소수점 8자리까지 일치)")


# ══════════════════════════════════════════════════════════════
# 데이터 로딩 — rpfi_eval.load_participant_data를 1회만 호출해 캐싱
# (detrend_tarvainen은 O(N^3) 행렬역산이라 파라미터 스윕 때마다 다시 돌리면 안 됨)
# ══════════════════════════════════════════════════════════════

def load_all_data(runs_dir, label_full, models, fs, seg_len, do_detrend, do_bpf, Lambda,
                  n_participants=None, n_chunks_per_participant=None, seed=0):
    rng = np.random.default_rng(seed)
    cache = {}
    for model in models:
        data = R.load_participant_data(runs_dir, label_full, model, fs, seg_len,
                                       do_detrend, do_bpf, Lambda)
        if not data:
            warnings.warn(f"{model}: load_participant_data 결과 없음 (경로/파일명 확인)")
            continue
        multi = {s: d["folds"] for s, d in data.items() if len(d["folds"]) > 1}
        if multi:
            warnings.warn(f"{model}: participant {len(multi)}명이 여러 fold에 걸쳐 나타남 "
                          f"— cross_dataset_eval.py 산출물을 잘못된 --runs로 채점 중일 수 있음. "
                          f"within-dataset 결과라면 --runs 경로를 다시 확인하세요.")
        subs = sorted(data)
        if n_participants and len(subs) > n_participants:
            subs = sorted(rng.choice(subs, n_participants, replace=False).tolist())
        subj_cache = {}
        for s in subs:
            d = data[s]
            n_chunk = len(d["label"])
            if n_chunk == 0:
                continue
            idx = (sorted(rng.choice(n_chunk, min(n_chunks_per_participant, n_chunk),
                                     replace=False).tolist())
                  if n_chunks_per_participant else list(range(n_chunk)))
            subj_cache[s] = dict(label=d["label"][idx], pred=d["pred"][idx],
                                 label_edd=d["label_edd"][idx], pred_edd=d["pred_edd"][idx])
        if subj_cache:
            cache[model] = subj_cache
            print(f"  [캐싱] {model}: participant {len(subj_cache)}명, "
                 f"참여자당 청크 {[len(v['label']) for v in subj_cache.values()]}")
    return cache


def comp_means_from_cache(cache, fs, params):
    """params: dict(lambda_p, lambda_a, dtw_band_frac, Sc, Ss[, wcr_beta])
    weight_sensitivity()와 동일한 순서: participant별 원값 -> 모델 단위 평균."""
    beta = params.get("wcr_beta", R.WCR_BETA)
    out = {}
    for model, subs_data in cache.items():
        comp_vals = {c: [] for c in R.COMPONENTS}
        for subj, d in sorted(subs_data.items()):
            L, P, L_e, P_e = d["label"], d["pred"], d["label_edd"], d["pred_edd"]
            if len(L) == 0:
                continue
            bwmd = float(np.mean([
                bwmd_param(L[c], P[c], fs, params["lambda_p"], params["lambda_a"],
                          params["dtw_band_frac"]) for c in range(len(L))]))
            y_cat, p_cat = L.ravel(), P.ravel()
            y_cat_e, p_cat_e = L_e.ravel(), P_e.ravel()
            sre = sre_param(y_cat, p_cat, fs, params["Sc"], params["Ss"])
            wcr, _, _ = R.compute_wcr(y_cat, p_cat, beta=beta)
            edd = R.compute_edd(y_cat_e, p_cat_e, fs)
            comp_vals["BWMD"].append(bwmd); comp_vals["SRE"].append(sre)
            comp_vals["WCR"].append(wcr); comp_vals["EDD"].append(edd)
        if comp_vals["BWMD"]:
            out[model] = {c: float(np.mean(v)) for c, v in comp_vals.items()}
    return out


def rpfi_ranking(comp_means, anchors, weights, agg):
    scores = {}
    for m, comp in comp_means.items():
        s = {c: R.normalize_component(c, comp[c], anchors) for c in R.COMPONENTS}
        scores[m] = R.aggregate(s, weights, agg)
    return scores


# ══════════════════════════════════════════════════════════════
# 스윕
# ══════════════════════════════════════════════════════════════

BASE_PARAMS = dict(lambda_p=R.BWMD_LAMBDA_P, lambda_a=R.BWMD_LAMBDA_A,
                   dtw_band_frac=0.10, Sc=R.SRE_SNR_LOW, Ss=R.SRE_SNR_HIGH,
                   wcr_beta=R.WCR_BETA)
GRIDS = {
    "dtw_band_frac": [0.05, 0.075, 0.10, 0.15, 0.20, 0.30],
    "lambda_p":      [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    "lambda_a":      [0.25, 0.375, 0.5, 0.625, 0.75, 1.0],
    "Sc":            [1.0, 2.0, 3.0, 4.0, 5.0],
    "Ss":            [10.0, 12.5, 15.0, 17.5, 20.0],
    "wcr_beta":      [0.3, 0.45, 0.6, 0.75, 0.9],   # 보너스: 이미 파라미터화돼있어 거의 공짜
}


def rank_stats(baseline, perturbed):
    common = sorted(set(baseline) & set(perturbed))
    a = [baseline[m] for m in common]
    b = [perturbed[m] for m in common]
    rho, _ = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    top1 = max(common, key=lambda m: baseline[m]) == max(common, key=lambda m: perturbed[m])
    last = min(common, key=lambda m: baseline[m]) == min(common, key=lambda m: perturbed[m])
    return dict(spearman_rho=round(float(rho), 4), kendall_tau=round(float(tau), 4),
               top1_retained=top1, last_retained=last, n_models=len(common))


def oat_sweep(cache, fs, param_name, anchors, weights, agg):
    baseline = rpfi_ranking(comp_means_from_cache(cache, fs, BASE_PARAMS), anchors, weights, agg)
    rows = []
    for v in GRIDS[param_name]:
        p = dict(BASE_PARAMS); p[param_name] = v
        scores = rpfi_ranking(comp_means_from_cache(cache, fs, p), anchors, weights, agg)
        stat = rank_stats(baseline, scores)
        stat.update({param_name: v, "is_baseline": v == BASE_PARAMS[param_name]})
        stat["scores"] = {m: round(s, 2) for m, s in scores.items()}
        rows.append(stat)
    return baseline, rows


def joint_random_sweep(cache, fs, anchors, weights, agg, n_trials, seed=0):
    rng = np.random.default_rng(seed)
    baseline = rpfi_ranking(comp_means_from_cache(cache, fs, BASE_PARAMS), anchors, weights, agg)
    rows = []
    for t in range(n_trials):
        p = dict(
            dtw_band_frac=float(rng.uniform(0.05, 0.30)),
            lambda_p=float(rng.uniform(0.5, 2.0)),
            lambda_a=float(rng.uniform(0.25, 1.0)),
            Sc=float(rng.uniform(1.0, 5.0)),
            Ss=float(rng.uniform(10.0, 20.0)),
            wcr_beta=float(rng.uniform(0.3, 0.9)),
        )
        scores = rpfi_ranking(comp_means_from_cache(cache, fs, p), anchors, weights, agg)
        stat = rank_stats(baseline, scores)
        stat["trial"] = t
        rows.append(stat)
    return baseline, rows


# ══════════════════════════════════════════════════════════════
# main — 인자 구성은 rpfi_eval.py의 CLI 관례를 그대로 따름
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True, help="runs/{dataset}_cv{K} 디렉토리")
    ap.add_argument("--label", required=True, help="데이터셋 전체 label npy")
    ap.add_argument("--models", default=None, help="콤마 구분. 미지정시 자동 탐색")
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--agg", choices=["geometric", "arithmetic"], default="geometric")
    ap.add_argument("--anchors", default=None,
                    help="derive_anchors.py 출력 JSON. 강력 권장 (§상단 경고 참고)")
    ap.add_argument("--weights", default=None, help="가중치 JSON (미지정시 rpfi_eval 기본값)")
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--param", choices=list(GRIDS), default=None,
                    help="OAT 스윕할 파라미터 1개. 미지정시 전부 순회")
    ap.add_argument("--joint-random", type=int, default=0,
                    help="지정하면 OAT 대신 전 파라미터 동시 무작위 스윕 N회")
    ap.add_argument("--n-participants", type=int, default=15,
                    help="속도용 서브샘플 (DTW가 순수 python). 0=전체")
    ap.add_argument("--n-chunks-per-participant", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="hyperparam_sensitivity_report")
    args = ap.parse_args()

    fs = args.fs or R.FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    print("[self-check] 파라미터화 래퍼 vs rpfi_eval 원본 대조 중...")
    _self_check(fs=30)

    anchors = dict(FALLBACK_ANCHORS)
    if args.anchors:
        anchors.update({k: tuple(v) for k, v in json.load(open(args.anchors)).items()})
        print(f"[anchor] {args.anchors} 사용")
    else:
        warnings.warn("--anchors 미지정 — 스크립트 내장 anchor(이전 세션에서 확정한 값)를 씁니다. "
                      "derive_anchors.py의 실제 출력 JSON을 --anchors로 넘기는 걸 권장합니다.")
    weights = dict(R.DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.load(open(args.weights)))
    print(f"[설정] anchors={anchors}")
    print(f"[설정] weights={weights}  agg={args.agg}")

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
             if args.models else R.discover_models(args.runs))
    if not models:
        raise SystemExit(f"{args.runs}에서 모델을 찾지 못했습니다.")
    print(f"[모델] {models}")

    label_full = np.load(args.label)
    print(f"\n[로딩] participant 단위 전처리 캐싱 중 (1회만 수행)...")
    cache = load_all_data(args.runs, label_full, models, fs, seg_len,
                          not args.no_detrend, not args.no_bpf, args.lambda_detrend,
                          args.n_participants or None, args.n_chunks_per_participant,
                          args.seed)
    if not cache:
        raise SystemExit("캐싱된 데이터가 없습니다. --runs/--label 경로를 확인하세요.")

    import pandas as pd

    if args.joint_random:
        print(f"\n=== 전 파라미터 동시 무작위 스윕 ({args.joint_random}회) ===")
        baseline, rows = joint_random_sweep(cache, fs, anchors, weights, args.agg,
                                            args.joint_random, args.seed)
        print(f"baseline 1위 모델: {max(baseline, key=baseline.get)}")
        df = pd.DataFrame(rows)
        print(df[["spearman_rho", "kendall_tau", "top1_retained", "last_retained"]]
             .agg(["mean", "std", "min"]).to_string())
        print(f"\ntop1_retained 비율: {df['top1_retained'].mean():.1%}")
        print(f"last_retained 비율: {df['last_retained'].mean():.1%}")
        df.to_csv(out_dir / f"joint_random_{args.dataset}.csv", index=False)
        print(f"\n(참고) §4-f 가중치 민감도: rho mean 0.95~0.99, top1/last 유지율 100% "
             f"— 이 결과와 나란히 표로 비교해서 보고")
    else:
        params_to_run = [args.param] if args.param else list(GRIDS)
        for pname in params_to_run:
            print(f"\n=== OAT 스윕: {pname} (baseline={BASE_PARAMS[pname]}) ===")
            baseline, rows = oat_sweep(cache, fs, pname, anchors, weights, args.agg)
            print(f"baseline 순위: " +
                 " > ".join(f"{m}({s:.1f})" for m, s in
                           sorted(baseline.items(), key=lambda x: -x[1])))
            df = pd.DataFrame(rows)
            print(df.drop(columns=["scores"]).to_string(index=False))
            df.to_csv(out_dir / f"oat_{pname}_{args.dataset}.csv", index=False)
            worst = df.loc[df["spearman_rho"].idxmin()]
            print(f"-> 가장 크게 흔든 값: {pname}={worst[pname]} "
                 f"(rho={worst['spearman_rho']}, top1_retained={worst['top1_retained']})")

    print(f"\n모든 산출물 저장: {out_dir}/")


if __name__ == "__main__":
    main()

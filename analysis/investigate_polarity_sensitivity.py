"""
investigate_polarity_sensitivity.py — 극성(부호) 반전이 RPFI 채점에 미치는 영향 정량화
====================================================================================
배경
----
lag_corrected_beat_template.py의 자체 진단(비트 단위, skewness 기반 극성검출)에서
physdiff(13~41%)와 POS(25~57%)가 다른 모델(대부분 <17%, 대부분 <9%)보다 훨씬 높은 비율로
"극성 반전이 필요하다"는 결과가 나왔다. 그런데 실제 채점 파이프라인(rpfi_eval.py)의
compute_bwmd/compute_wcr/compute_edd 어디에도 극성보정 로직이 없다:
  - WCR(CCC+Spearman)은 부호에 민감 — 신호를 뒤집으면 상관계수가 강한 음수가 된다.
  - EDD(잔차의 심장대역 파워비율)도 극성반전 시 잔차(yhat-y)가 대략 -2y를 따라가게 돼
    잔차 에너지가 심장대역에 집중되어 EDD가 나빠진다(lower-is-better인데 값이 커짐).
  - BWMD의 피크검출(peak_times)도 반전되면 피크 대신 골을 검출할 수 있어 악영향 가능.
즉 physdiff/POS가 실제로 극성반전된 출력을 자주 낸다면, 이 두 모델의 낮은 점수 일부가
"파형 품질 자체가 나빠서"가 아니라 "부호가 뒤집혀서 나쁘게 채점된 것"일 가능성이 있다.

이 스크립트가 하는 일 (공식 파이프라인은 건드리지 않는다)
--------------------------------------------------------
rpfi_eval.py와 완전히 동일한 전처리/컴포넌트 계산 함수를 그대로 재사용해서, *실제 평가에
쓰인 청크*(lag_corrected_beat_template.py의 자체 비트 윈도우가 아니라, Table I/cross-dataset
결과를 만든 것과 동일한 10초 청크) 기준으로 극성반전율을 직접 측정하고, "만약 청크 단위로
극성을 정렬한 뒤 채점했다면 BWMD/WCR/EDD/RPFI가 얼마나 달라졌을까"를 계산한다.

극성 검출 방법: 청크별로 label과 pred의 Pearson 상관계수 부호를 본다(corr<0 이면 반전
필요로 판정). 이는 lag_corrected_beat_template.py의 비트 단위 skewness 검출과는 검출
단위(청크 10초 vs 비트 1개)와 방법이 다르므로, 두 스크립트의 반전율 숫자를 곧바로
같은 것으로 비교하면 안 된다 — 이 스크립트는 "실제 채점 청크 기준으로 다시 재보면
비슷한 정도로 나오는지, 그리고 그게 점수에 실제로 영향을 주는지"를 독립적으로 확인한다.

★ 주의 — 이 보정은 "오라클(정답을 알고 사후에 부호를 맞추는)" 방식이다.
label과 비교해서 pred의 부호를 정하므로, 실사용 가능한 실시간 보정이 아니라 어디까지나
"극성이 문제의 본질인지 확인하는" 진단용 상한선(best-case) 추정이다.
  - 극성정렬 후에도 physdiff/POS 점수가 별로 안 바뀌면 → 극성은 원인이 아니라는 뜻
    (Limitations에 캐주얼하게 한 줄 남길 필요도 없어짐 — 오히려 "확인해봤지만 아니었다"로
    더 깔끔하게 닫을 수 있음).
  - 크게 개선되면 → 극성이 실제로 원인이라는 근거는 확보되지만, "실사용 가능한 보정법"
    (예: 학습 시 부호를 고정하는 손실 추가, 또는 평가 시 비지도 극성정렬 방법 도입)은
    이 스크립트의 범위 밖이며 별도 논의가 필요.

이 스크립트는 rpfi_eval.py를 수정하지 않고, 이미 논문에 반영된 어떤 수치도 이 스크립트
실행만으로는 바뀌지 않는다 — hyperparam_sensitivity.py/investigate_physdiff_asymmetry.py와
같은 성격의 부가 민감도 분석 스크립트다.

[2026-09-28] v2 수정 — 상관계수 크기(magnitude) 임계값 추가
------------------------------------------------------------
v1은 corr<0이면 무조건 "반전 필요"로 판정했다. 문제: 이러면 진짜 극성반전(부호가 뒤집힌
좋은 신호)과 그냥 잡음 때문에 상관계수가 0 근처에서 우연히 음수로 나온 저품질 청크를
구분하지 못한다. 실제 3개 데이터셋 실행 결과에서 이 문제가 드러났다 — 재구성 품질이
가장 나쁜 COHFACE에서 사실상 전 모델(physdiff 포함, 30.9~38.4%)이 거의 균일하게 높은
"반전율"을 보였고, 재구성 품질이 가장 좋은 PURE에서는 전 모델이 낮은 반전율(3~7%)을
보였다 — 이는 모델별 극성 컨벤션 문제라기보다 "상관계수가 약할수록 부호가 잡음에 의해
우연히 뒤집힐 확률이 높아진다"는 confound와 정확히 일치하는 패턴이다. 즉 v1의 반전율/
ΔRPFI 수치는 진짜 극성문제와 저품질-청크-잡음이 뒤섞여 있어 그대로 논문에 쓰기에는
근거가 약하다.
수정: `--min-abs-corr` 임계값을 추가해 "반전 필요"를 `corr <= -threshold`로만 판정하고
(약한 음의 상관은 더 이상 자동으로 반전 플래그를 받지 않음), 추가로 여러 임계값
(0.0/0.1/0.2/0.3/0.4)에서의 반전율을 모델별로 한 번에 출력하는 민감도 표를 추가했다.
임계값을 올렸을 때 반전율이 급격히 떨어지면 원래 신호가 대부분 noise-confound였다는
뜻이고, 임계값을 올려도 특정 모델(들)만 여전히 높게 남으면 그건 진짜 신호일 가능성이
높다. 실제 오라클 재채점(``score_participant``)은 CLI로 지정한 `--min-abs-corr` 값
하나로 수행되며 기본값은 0.0(=v1과 동일, 하위호환)이다 — 논문에 반영하기 전에는 반드시
threshold>0인 결과의 민감도 표부터 확인할 것.

사용법
------
    python investigate_polarity_sensitivity.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" --anchors anchors/rpfi_anchors.json \
        --out polarity_sensitivity --min-abs-corr 0.2

    # 세 데이터셋 다 확인하려면 --dataset만 바꿔서 3번 실행
    # --min-abs-corr을 지정하지 않으면 0.0(v1과 동일) — 콘솔에 항상 출력되는
    # "임계값별 반전율 민감도 표"로 confound 여부를 먼저 확인하고 값을 정할 것
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from rpfi_eval import (
    FS_BY_DATASET, COMPONENTS, DEFAULT_WEIGHTS,
    discover_models, load_participant_data,
    compute_bwmd, compute_wcr, compute_edd,
    normalize_component, aggregate,
)


# ══════════════════════════════════════════════════════════════
# 극성 검출 + 보정 채점
# ══════════════════════════════════════════════════════════════

def chunk_correlations(label_chunks, pred_chunks):
    """청크별 label-pred Pearson 상관계수를 그대로 반환 (분산 0인 청크는 NaN).
    민감도 표(여러 threshold)와 실제 flip 판정 양쪽에서 재사용한다."""
    corrs = []
    for l, p in zip(label_chunks, pred_chunks):
        if np.std(l) < 1e-12 or np.std(p) < 1e-12:
            corrs.append(np.nan)
            continue
        corrs.append(np.corrcoef(l, p)[0, 1])
    return np.array(corrs, dtype=float)


def chunk_polarity_flags(corrs, min_abs_corr=0.0):
    """상관계수 배열에서 '반전 필요' 청크를 판정한다.
    True 조건: corr <= -min_abs_corr (NaN/약한 음의 상관은 반전으로 판정하지 않음).
    min_abs_corr=0.0이면 v1과 동일(corr<0이면 전부 반전)."""
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(corrs), False, corrs <= -min_abs_corr)


def corr_sensitivity_table(corrs, thresholds=(0.0, 0.1, 0.2, 0.3, 0.4)):
    """임계값별 (반전 판정률, 애매해서 제외되는 비율)을 계산 — noise-confound 진단용."""
    valid = ~np.isnan(corrs)
    n_valid = int(valid.sum())
    out = []
    for t in thresholds:
        flip = corrs <= -t
        flip_rate = float(np.sum(flip & valid)) / max(n_valid, 1)
        ambiguous = float(np.sum((np.abs(corrs) < t) & valid)) / max(n_valid, 1)
        out.append((t, flip_rate, ambiguous))
    return out


def score_participant(L, P, L_e, P_e, fs, anchors, weights, agg_mode, flip_mask=None):
    """rpfi_eval.py의 participant 처리와 동일한 절차.
    flip_mask가 주어지면 해당 청크의 pred/pred_edd 부호를 반전한 뒤 채점한다
    (전처리 자체는 이미 detrend+BPF/no-BPF+z-norm까지 끝난 상태라, 부호만 뒤집는 건
    z-norm의 평균0/분산1 성질을 깨지 않는 잘 정의된 사후 연산이다)."""
    if flip_mask is not None and flip_mask.any():
        P = P.copy()
        P_e = P_e.copy()
        P[flip_mask] *= -1.0
        P_e[flip_mask] *= -1.0

    bwmds = [compute_bwmd(L[c], P[c], fs)[0] for c in range(len(L))]
    bwmd = float(np.mean(bwmds))
    y, p = L.ravel(), P.ravel()
    wcr = compute_wcr(y, p)[0]
    edd = compute_edd(L_e.ravel(), P_e.ravel(), fs)
    comp = {"BWMD": bwmd, "WCR": wcr, "EDD": edd}
    scores = {c: normalize_component(c, comp[c], anchors) for c in COMPONENTS}
    rpfi = aggregate(scores, weights, agg_mode)
    return rpfi, comp


# ══════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--agg", choices=["geometric", "arithmetic"], default="geometric")
    ap.add_argument("--models", default=None, help="콤마 구분. 미지정시 전체")
    ap.add_argument("--out", default="polarity_sensitivity")
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--min-abs-corr", type=float, default=0.0,
                     help="반전 판정에 쓰는 |corr| 최소 임계값. 0.0=v1과 동일(모든 음의 상관을 "
                          "반전으로 판정). 콘솔에 항상 출력되는 임계값별 민감도 표를 먼저 보고 "
                          "정할 것 — noise-confound가 의심되면 0.2~0.4 권장.")
    args = ap.parse_args()

    fs = args.fs or FS_BY_DATASET[args.dataset]
    seg_len = int(round(fs * args.eval_seconds))
    do_detrend, do_bpf = not args.no_detrend, not args.no_bpf

    anchors = {k: tuple(v) for k, v in json.load(open(args.anchors)).items()}
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
    print(f"극성반전 민감도 분석 — dataset={args.dataset}  fs={fs}Hz  models={models}")
    print(f"  (반전 판정 기준: corr <= -{args.min_abs_corr:.2f}. lag_corrected_beat_template.py의")
    print("   비트 단위 skewness 기준과는 검출 단위가 다르므로 반전율 수치를 직접 비교하지 말 것)")
    print("=" * 78)

    rows = []
    per_model_base, per_model_corrected = {}, {}
    inversion_rate_rows = []
    all_corrs_by_model = {}

    for model in models:
        data = load_participant_data(args.runs, label_full, model, fs, seg_len,
                                     do_detrend, do_bpf, args.lambda_detrend)
        if not data:
            print(f"  [건너뜀] {model}: 산출물 없음")
            continue

        base_rpfis, corr_rpfis = {}, {}
        n_chunks_total, n_flip_total = 0, 0
        model_corrs = []

        for subj, d in sorted(data.items()):
            L, P = d["label"], d["pred"]
            L_e, P_e = d["label_edd"], d["pred_edd"]
            if len(L) == 0:
                continue

            corrs = chunk_correlations(L, P)
            flip_mask = chunk_polarity_flags(corrs, args.min_abs_corr)
            model_corrs.append(corrs)
            n_chunks_total += len(flip_mask)
            n_flip_total += int(flip_mask.sum())

            rpfi_base, comp_base = score_participant(
                L, P, L_e, P_e, fs, anchors, weights, args.agg, flip_mask=None)
            rpfi_corr, comp_corr = score_participant(
                L, P, L_e, P_e, fs, anchors, weights, args.agg, flip_mask=flip_mask)
            base_rpfis[subj] = rpfi_base
            corr_rpfis[subj] = rpfi_corr

            rows.append(dict(
                dataset=args.dataset, model=model, participant=subj,
                min_abs_corr=args.min_abs_corr,
                n_chunks=len(flip_mask), n_flipped=int(flip_mask.sum()),
                flip_rate=float(flip_mask.mean()),
                RPFI_base=rpfi_base, RPFI_polarity_corrected=rpfi_corr,
                delta_RPFI=rpfi_corr - rpfi_base,
                BWMD_base=comp_base["BWMD"], BWMD_corrected=comp_corr["BWMD"],
                WCR_base=comp_base["WCR"], WCR_corrected=comp_corr["WCR"],
                EDD_base=comp_base["EDD"], EDD_corrected=comp_corr["EDD"]))

        per_model_base[model] = base_rpfis
        per_model_corrected[model] = corr_rpfis
        all_corrs_by_model[model] = np.concatenate(model_corrs) if model_corrs else np.array([])
        flip_rate_model = n_flip_total / max(n_chunks_total, 1)
        inversion_rate_rows.append((model, n_chunks_total, n_flip_total, flip_rate_model))

        mean_delta = float(np.mean([corr_rpfis[s] - base_rpfis[s] for s in base_rpfis]))
        print(f"  [{model:16s}] 청크 {n_chunks_total:5d}개 중 반전 {n_flip_total:5d}개 "
              f"({100*flip_rate_model:5.1f}%)   평균 ΔRPFI(보정-원본) = {mean_delta:+.2f}")

    # ── 모델별 청크단위 반전율 요약 (현재 --min-abs-corr 기준) ──
    print("\n" + "=" * 78)
    print(f"청크 단위(corr<=-{args.min_abs_corr:.2f}) 극성반전율 — 모델별")
    print("=" * 78)
    for model, n_tot, n_flip, rate in sorted(inversion_rate_rows, key=lambda x: -x[3]):
        flag = "  ⚠" if rate > 0.05 else ""
        print(f"  {model:16s}: {n_flip:5d}/{n_tot:5d}  ({100*rate:5.1f}%){flag}")

    # ── 임계값별 민감도 표: noise-confound 여부 진단 ──
    print("\n" + "=" * 78)
    print("임계값별 반전율 민감도 (noise-confound 확인용 — |corr|<threshold 청크는 '애매'로 제외)")
    print("=" * 78)
    thresholds = (0.0, 0.1, 0.2, 0.3, 0.4)
    header = "  " + f"{'model':16s}" + "".join(f"  t={t:.1f}(flip/ambig)" for t in thresholds)
    print(header)
    for model in sorted(all_corrs_by_model, key=lambda m: -inversion_rate_rows[
            [r[0] for r in inversion_rate_rows].index(m)][3]):
        corrs = all_corrs_by_model[model]
        cells = corr_sensitivity_table(corrs, thresholds)
        line = f"  {model:16s}"
        for t, flip_rate, ambiguous in cells:
            line += f"   {100*flip_rate:4.1f}/{100*ambiguous:4.1f}%"
        print(line)
    print("  (해석: threshold를 올려도 특정 모델만 flip%가 높게 남으면 진짜 신호;")
    print("   전 모델이 비슷하게 낮아지면 t=0.0 반전율은 대부분 noise-confound였다는 뜻)")

    # ── 순위 변화: 원본 vs 극성보정 ──
    common_models = [m for m in per_model_base if per_model_base[m]]
    subs_common = sorted(set.intersection(*[set(per_model_base[m]) for m in common_models])) \
        if len(common_models) >= 2 else (
            sorted(per_model_base[common_models[0]]) if common_models else [])

    def model_mean(d, subs):
        return {m: float(np.mean([d[m][s] for s in subs])) for m in d if subs}

    base_means = model_mean(per_model_base, subs_common)
    corr_means = model_mean(per_model_corrected, subs_common)

    if len(common_models) >= 2 and subs_common:
        base_rank = sorted(base_means, key=lambda m: -base_means[m])
        corr_rank = sorted(corr_means, key=lambda m: -corr_means[m])
        base_vals = [base_means[m] for m in common_models]
        corr_vals = [corr_means[m] for m in common_models]
        rho, _ = spearmanr(base_vals, corr_vals)

        print("\n" + "=" * 78)
        print(f"모델 순위: 원본 vs 극성보정 (공통 participant N={len(subs_common)} 평균 RPFI 기준)")
        print("=" * 78)
        print(f"  {'model':16s} {'RPFI(원본)':>12s} {'RPFI(보정)':>12s} {'Δ':>8s}")
        for m in sorted(common_models, key=lambda x: -base_means[x]):
            print(f"  {m:16s} {base_means[m]:12.2f} {corr_means[m]:12.2f} "
                  f"{corr_means[m]-base_means[m]:+8.2f}")
        print(f"\n  기준 1위(원본): {base_rank[0]}   극성보정 후 1위: {corr_rank[0]}")
        print(f"  기준 꼴찌(원본): {base_rank[-1]}   극성보정 후 꼴찌: {corr_rank[-1]}")
        print(f"  순위 Spearman ρ(원본 vs 보정): {rho:.4f}")
        if base_rank[0] != corr_rank[0]:
            print("  ⚠ 극성보정으로 1위 모델이 바뀜 — 극성이 순위에 실질적 영향을 준다는 뜻")
        else:
            print("  1위 모델 유지됨")
        if base_rank[-1] != corr_rank[-1]:
            print("  ⚠ 극성보정으로 꼴찌 모델이 바뀜")
        else:
            print("  꼴찌 모델 유지됨")
    else:
        print("\n  모델 수 부족으로 순위비교 생략")

    # ── CSV 저장 ──
    f1 = out_dir / f"polarity_sensitivity_{args.dataset}.csv"
    if rows:
        with open(f1, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n저장: {f1}")
    else:
        print("\n⚠ 기록된 행이 없습니다 (데이터 로드 확인 필요).")


if __name__ == "__main__":
    main()

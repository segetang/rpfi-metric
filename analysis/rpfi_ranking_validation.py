"""
rpfi_ranking_validation.py — RPFI 순위 복원 정량 검증 (Kendall's τ)
=====================================================================
"RPFI가 순위를 정말 잘 뽑느냐"에 대한 정량적 답 (R1-8 보강, 갤러리의 정성적
증거를 정량적 통계로 뒷받침).

방법
----
매 trial마다:
  1) 무작위 참조 PPG 하나를 만든다.
  2) 그 참조에 열화를 강도 0(무열화, 객관적 최선) ~ 1(최대, 객관적 최악)까지
     K_systems단계로 적용해 "품질이 이미 알려진 순서대로인" 가상 시스템 K개를
     만든다. 진짜 순위는 강도로 정의되므로 100% 확실하다(순환논리 없음).
  3) 각 시스템의 RPFI를 계산하고, "강도 순서"와 "RPFI 순서"가 얼마나 일치하는지
     Kendall's τ로 잰다(τ=1: 완벽히 순서대로 재현, τ=0: 무작위, τ=-1: 정반대).
  4) 수백~수천 trial 반복(참조신호·열화종류·fs 무작위)해서 τ의 분포를 낸다.

두 가지 trial 모드
-------------------
  single — 한 trial에 열화종류 1개만 강도별로 적용. 가장 깨끗한 ground truth.
  mixed  — 2~4개 열화종류를 동시에, 공통 강도 파라미터로 같이 스케일.
           "복합적으로 나쁜 시스템"에 더 가까운 현실적 시뮬레이션이지만,
           진짜 순위는 여전히 공통 강도 하나로 명확히 정의됨(순환논리 없음).

rpfi_eval.py / derive_anchors.py와 동일한 함수를 그대로 import해서 쓰므로
anchor/가중치/집계 방식이 실제 평가와 완전히 동일하다.

사용법
------
    python rpfi_ranking_validation.py --anchors anchors/rpfi_anchors.json \
        --n-trials 2000 --out ranking_validation
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

import rpfi_eval
from rpfi_eval import COMPONENTS, normalize_component, aggregate, DEFAULT_WEIGHTS
from derive_anchors import DEGRADATIONS, make_case, components_for_case

# 평가경로의 z-norm(진폭·오프셋 제거, R1-4)과 detrend(저주파 wander 제거)가
# "설계상 의도적으로 안 보이게" 만드는 열화들. 이 축은 RPFI가 둔감한 게 버그가
# 아니라 의도이므로, 순위복원 정확도를 이 축까지 섞어서 하나의 숫자로 내면
# 오해를 부른다. summary를 "전체" / "의미있는 열화만"으로 나눠 보고한다.
INVARIANT_DEGRADATIONS = {"amplitude_scale", "dc_offset", "baseline_wander"}


def run_trial(rng, fs, seg_len, n_chunks, k_systems, mode,
              do_detrend, do_bpf, Lambda, anchors, weights, agg_mode):
    severities = np.linspace(0.0, 1.0, k_systems)  # 0=무열화(최선) ... 1=최대(최악)

    if mode == "single":
        dname = rng.choice(list(DEGRADATIONS))
        dfns = [DEGRADATIONS[dname]]
        tag = dname
        names_used = [dname]
    else:
        names = list(DEGRADATIONS)
        n_mix = int(rng.integers(2, 5))
        chosen = list(rng.choice(names, size=n_mix, replace=False))
        dfns = [DEGRADATIONS[d] for d in chosen]
        tag = "+".join(chosen)
        names_used = chosen

    all_invariant = all(n in INVARIANT_DEGRADATIONS for n in names_used)

    rpfis = []
    for s in severities:
        def tf(y, fs_, r, hr, _fns=dfns, _s=s):
            z = y.copy()
            for f in _fns:
                z = f(z, fs_, _s, r)
            return z
        ref, pred = make_case(fs, seg_len, n_chunks, rng, tf)
        comp = components_for_case(ref, pred, fs, do_detrend, do_bpf, Lambda)
        scores = {c: normalize_component(c, comp[c], anchors) for c in COMPONENTS}
        rpfis.append(aggregate(scores, weights, agg_mode))

    rpfis = np.asarray(rpfis)
    # 진짜 순위(강도 낮을수록 좋음)와 RPFI 순위가 일치하면 τ→+1이 되도록 부호 정리
    tau, _ = kendalltau(-severities, rpfis)
    perfect = bool(np.all(np.diff(rpfis) <= 1e-9))  # 강도 증가 = RPFI 비증가(완벽 단조)
    return dict(mode=mode, tag=tag, fs=fs, tau=float(tau) if not np.isnan(tau) else 0.0,
               perfect=perfect, all_invariant=all_invariant,
               rpfis=rpfis.tolist(), severities=severities.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--agg", choices=["geometric", "arithmetic"], default="geometric")
    ap.add_argument("--n-trials", type=int, default=2000, help="모드당 trial 수")
    ap.add_argument("--k-systems", type=int, default=5, help="trial당 비교할 시스템 수")
    ap.add_argument("--fs-list", default="30,20")
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--n-chunks", type=int, default=4, help="가상 participant당 chunk 수")
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ranking_validation")
    ap.add_argument("--dtw-backend", choices=["naive", "fastdtw"], default="naive",
                    help="naive가 기본값. seq_len 200~300 같은 짧은 신호에서는 "
                        "fastdtw 패키지의 파이썬 오버헤드가 커서 naive Sakoe-Chiba가 "
                        "실측 약 10배 더 빠름(1.5s vs 14s/trial). rpfi_eval.py 실제 평가처럼 "
                        "긴 신호를 쓸 땐 --dtw-backend fastdtw로 바꿀 것.")
    args = ap.parse_args()
    rpfi_eval.USE_FASTDTW = (args.dtw_backend == "fastdtw")

    anchors = {k: tuple(v) for k, v in json.load(open(args.anchors)).items()}
    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        weights.update(json.load(open(args.weights)))
    fs_list = [int(x) for x in args.fs_list.split(",")]
    do_detrend, do_bpf = not args.no_detrend, not args.no_bpf
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("RPFI 순위 복원 검증 (Kendall's τ)")
    print("=" * 78)
    print(f"  trial당 시스템 수(K)={args.k_systems}  모드당 trial={args.n_trials}  "
          f"fs={fs_list}  가중치={weights}  집계={args.agg}")

    all_rows = []
    for mode in ["single", "mixed"]:
        print(f"\n[{mode}] {args.n_trials} trial 실행 중...")
        for i in range(args.n_trials):
            fs = int(rng.choice(fs_list))
            seg_len = int(round(fs * args.eval_seconds))
            r = run_trial(rng, fs, seg_len, args.n_chunks, args.k_systems, mode,
                         do_detrend, do_bpf, args.lambda_detrend,
                         anchors, weights, args.agg)
            all_rows.append(r)
            if (i + 1) % max(1, args.n_trials // 5) == 0:
                print(f"    {i + 1}/{args.n_trials}")

    # ── 요약 ──
    print("\n" + "=" * 78)
    print("결과 요약")
    print("=" * 78)
    for mode in ["single", "mixed"]:
        rows_m = [r for r in all_rows if r["mode"] == mode]
        taus = np.array([r["tau"] for r in rows_m])
        perfect_rate = np.mean([r["perfect"] for r in rows_m])
        print(f"\n>> {mode}  (n={len(taus)}, 진폭/오프셋/wander 포함 전체)")
        print(f"   Kendall τ: mean={taus.mean():.4f}  median={np.median(taus):.4f}  "
              f"std={taus.std():.4f}  min={taus.min():.4f}")
        print(f"   τ=1.0(완벽 순서 재현) 비율: {100 * np.mean(taus >= 0.999):.1f}%")
        print(f"   τ>=0.8 비율: {100 * np.mean(taus >= 0.8):.1f}%")
        print(f"   τ<0(역전) 비율: {100 * np.mean(taus < 0):.1f}%")
        print(f"   완전단조 비율: {100 * perfect_rate:.1f}%")

        rows_mean = [r for r in rows_m if not r["all_invariant"]]
        taus2 = np.array([r["tau"] for r in rows_mean])
        if len(taus2):
            print(f"   -- {mode} (amplitude_scale/dc_offset/baseline_wander만으로 "
                  f"구성된 trial 제외, n={len(taus2)}) --")
            print(f"   Kendall τ: mean={taus2.mean():.4f}  median={np.median(taus2):.4f}  "
                  f"min={taus2.min():.4f}")
            print(f"   τ>=0.8 비율: {100 * np.mean(taus2 >= 0.8):.1f}%")

    # single 모드: 열화종류별 breakdown (어떤 열화가 RPFI 순위를 못 살리는지)
    print("\n" + "=" * 78)
    print("[single] 열화종류별 τ (낮은 순 = RPFI가 순위를 못 살리는 열화)")
    print("=" * 78)
    by_tag = defaultdict(list)
    for r in all_rows:
        if r["mode"] == "single":
            by_tag[r["tag"]].append(r["tau"])
    rows_sorted = sorted(by_tag.items(), key=lambda x: np.mean(x[1]))
    for tag, taus in rows_sorted:
        taus = np.array(taus)
        print(f"  {tag:<20s} n={len(taus):4d}  mean τ={taus.mean():+.4f}  "
              f"min τ={taus.min():+.4f}  τ<0.8 비율={100 * np.mean(taus < 0.8):.1f}%")

    # 최악의 trial 몇 개 상세 출력 (실패 사례 진단용)
    worst = sorted(all_rows, key=lambda r: r["tau"])[:5]
    print("\n" + "=" * 78)
    print("τ가 가장 낮았던 trial 5개 (실패 사례 진단)")
    print("=" * 78)
    for r in worst:
        print(f"  mode={r['mode']:<7s} tag={r['tag']:<25s} fs={r['fs']:>2d}  τ={r['tau']:+.4f}")
        print(f"    severities={[f'{s:.2f}' for s in r['severities']]}")
        print(f"    RPFIs     ={[f'{v:.2f}' for v in r['rpfis']]}")

    # ── 저장 ──
    csv_path = out_dir / "ranking_validation_trials.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mode", "tag", "fs", "tau", "perfect", "all_invariant"] +
                  [f"severity_{i}" for i in range(args.k_systems)] +
                  [f"rpfi_{i}" for i in range(args.k_systems)])
        for r in all_rows:
            w.writerow([r["mode"], r["tag"], r["fs"], r["tau"], r["perfect"], r["all_invariant"]] +
                      r["severities"] + r["rpfis"])

    summary = {}
    for mode in ["single", "mixed"]:
        rows_m = [r for r in all_rows if r["mode"] == mode]
        taus = np.array([r["tau"] for r in rows_m])
        rows_mean = [r for r in rows_m if not r["all_invariant"]]
        taus2 = np.array([r["tau"] for r in rows_mean]) if rows_mean else taus
        summary[mode] = dict(n=len(taus), mean_tau=float(taus.mean()),
                             median_tau=float(np.median(taus)),
                             min_tau=float(taus.min()),
                             pct_perfect=float(100 * np.mean(taus >= 0.999)),
                             pct_tau_ge_0_8=float(100 * np.mean(taus >= 0.8)),
                             pct_reversed=float(100 * np.mean(taus < 0)),
                             n_excl_invariant=len(taus2),
                             mean_tau_excl_invariant=float(taus2.mean()),
                             pct_tau_ge_0_8_excl_invariant=float(100 * np.mean(taus2 >= 0.8)))
    json.dump(dict(config=vars(args), summary=summary),
              open(out_dir / "ranking_validation_summary.json", "w"), indent=2, default=str)

    print(f"\n저장: {csv_path}")
    print(f"저장: {out_dir / 'ranking_validation_summary.json'}")


if __name__ == "__main__":
    main()

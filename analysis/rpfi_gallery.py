"""
rpfi_gallery.py — RPFI 점수대별 실제 파형 시각 갤러리
=========================================================
목적: "RPFI가 X점이면 실제로 어떻게 생겼나"를 논문 Figure로 보여준다.
리뷰어의 "RPFI 30이 뭘 의미하냐" 질문에 대한 직관적(정성적) 답변.
(정량적 답변은 이미 확보: RPFI-HR_MAE 대응표, criterion validity r=-0.610)

두 가지 모드
------------
[quantile] 데이터셋 전체 (model × participant) 기록의 RPFI 분포에서
           지정한 분위수(기본 10/50/90)에 가장 가까운 실제 기록을 하나씩 골라
           나란히 그린다. "이 점수대는 대략 이렇게 생겼다"를 보여준다.

[subject]  RPFI 스프레드가 제일 큰 participant 한 명을 골라, 그 사람의
           **동일한 참조 신호**에 대해 여러 모델의 재구성을 나란히 그린다.
           참가자(=난이도)를 고정하고 모델만 바꾼 비교라, "같은 사람인데
           왜 모델마다 RPFI가 다른가"를 시각적으로 직접 보여준다.

전처리에 관한 투명성 원칙
--------------------------
그림에 쓰는 신호는 **BWMD/WCR/SRE가 실제로 채점한 것과 동일한 전처리**
(detrend→BPF→z-norm)를 거친 신호다. 원신호가 아니라 채점된 신호를 보여줘야
"이 점수가 왜 이렇게 나왔는지"를 정직하게 설명할 수 있다.

사용법
------
    python rpfi_gallery.py --dataset ubfc --runs runs/ubfc_cv5 \
        --label "../real data/UBFC_label.npy" \
        --anchors anchors/rpfi_anchors.json --out gallery

산출물
------
    gallery/{dataset}_quantile_gallery.png
    gallery/{dataset}_subject_gallery.png
    gallery/{dataset}_gallery_meta.csv   ← 그림에 쓴 정확한 participant/model/RPFI/컴포넌트값
                                            (캡션 작성 및 재현용)
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rpfi_eval import (
    FS_BY_DATASET, COMPONENTS, DEFAULT_WEIGHTS,
    discover_models, load_participant_data,
    compute_bwmd, compute_sre, compute_wcr, compute_edd,
    normalize_component, aggregate,
)


# ══════════════════════════════════════════════════════════════
# 데이터 수집: dataset 전체 (model × participant) RPFI + 원본 chunk
# ══════════════════════════════════════════════════════════════

def compute_all_records(runs_dir, label_full, models, fs, seg_len,
                        do_detrend, do_bpf, Lambda, anchors, weights, agg_mode):
    records = []
    for model in models:
        data = load_participant_data(runs_dir, label_full, model, fs, seg_len,
                                     do_detrend, do_bpf, Lambda)
        for subj, d in sorted(data.items()):
            L, P = d["label"], d["pred"]
            L_e, P_e = d["label_edd"], d["pred_edd"]
            if len(L) == 0:
                continue
            bwmds = [compute_bwmd(L[c], P[c], fs)[0] for c in range(len(L))]
            bwmd = float(np.mean(bwmds))
            y, p = L.ravel(), P.ravel()
            sre = compute_sre(y, p, fs)[0]
            wcr = compute_wcr(y, p)[0]
            edd = compute_edd(L_e.ravel(), P_e.ravel(), fs)
            comp = {"BWMD": bwmd, "SRE": sre, "WCR": wcr, "EDD": edd}
            scores = {c: normalize_component(c, comp[c], anchors) for c in COMPONENTS}
            rpfi = aggregate(scores, weights, agg_mode)
            records.append(dict(model=model, participant=subj, RPFI=rpfi, comp=comp,
                                label_chunks=L, pred_chunks=P, n_chunks=len(L)))
    return records


# ══════════════════════════════════════════════════════════════
# 예시 선택
# ══════════════════════════════════════════════════════════════

def select_quantile_examples(records, quantiles):
    rpfis = np.array([r["RPFI"] for r in records])
    used, chosen = set(), []
    for q in quantiles:
        target = np.percentile(rpfis, q)
        order = np.argsort(np.abs(rpfis - target))
        for idx in order:
            if idx not in used:
                used.add(int(idx))
                chosen.append((q, records[idx]))
                break
    return chosen


def select_subject_examples(records, n_show):
    """RPFI 스프레드가 가장 큰 participant를 고르고, 그 사람의 모델들 중
    상위 n_show개 + 중간 1개 + 하위 n_show개를 반환 (RPFI 내림차순 정렬)."""
    by_subj = defaultdict(list)
    for r in records:
        by_subj[r["participant"]].append(r)

    best_subj, best_spread = None, -1.0
    for subj, rs in by_subj.items():
        if len(rs) < max(3, 2 * n_show):
            continue
        spread = max(x["RPFI"] for x in rs) - min(x["RPFI"] for x in rs)
        if spread > best_spread:
            best_spread, best_subj = spread, subj
    if best_subj is None:
        return None, []

    rs = sorted(by_subj[best_subj], key=lambda x: -x["RPFI"])
    n = len(rs)
    mid = n // 2
    idxs = list(range(0, min(n_show, n))) + [mid] + \
        list(range(max(n - n_show, n_show), n))
    idxs = sorted(set(i for i in idxs if 0 <= i < n))
    return best_subj, [(None, rs[i]) for i in idxs]


# ══════════════════════════════════════════════════════════════
# 플롯
# ══════════════════════════════════════════════════════════════

def select_chunk_indices(n_total, n_show):
    """전체 n_total개 chunk 중 n_show개를 고르게 퍼뜨려 선택 (연속 X, 대표성 확보).
    n_total <= n_show면 전부 반환."""
    n_show = min(n_show, n_total)
    if n_total <= n_show:
        return list(range(n_total))
    idx = sorted(set(int(round(i)) for i in np.linspace(0, n_total - 1, n_show)))
    i = n_total - 1
    while len(idx) < n_show and i >= 0:      # linspace 중복 제거로 모자라면 뒤에서 채움
        if i not in idx:
            idx.append(i)
        i -= 1
    return sorted(idx)[:n_show]


def plot_gallery(examples, fs, seg_len, out_path, n_chunks_show, title):
    n = len(examples)
    fig, axs = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=False)
    axs = [axs] if n == 1 else list(axs)

    for ax, (q, r) in zip(axs, examples):
        idxs = select_chunk_indices(r["n_chunks"], n_chunks_show)
        y = np.concatenate([r["label_chunks"][i] for i in idxs])
        p = np.concatenate([r["pred_chunks"][i] for i in idxs])
        t = np.arange(len(y)) / fs
        seg_dur = seg_len / fs

        ax.plot(t, y, color="black", lw=1.3, label="Reference")
        ax.plot(t, p, color="#d62728", lw=1.1, alpha=0.85, label="Prediction")
        for c in range(1, len(idxs)):
            ax.axvline(c * seg_dur, color="gray", lw=0.5, ls=":")

        # 각 segment가 원래 recording의 몇 초 지점인지 라벨 (연속 구간이 아님을 명시)
        for c, ci in enumerate(idxs):
            ax.text((c + 0.5) * seg_dur, 0.96, f"orig t={ci * seg_dur:.0f}s",
                    ha="center", va="top", fontsize=6.5, color="gray",
                    transform=ax.get_xaxis_transform())

        qtag = f"p{q}  " if q is not None else ""
        total_sec = r["n_chunks"] * seg_dur
        coverage = (f"[{len(idxs)} chunks spread across full {total_sec:.0f}s "
                   f"({len(idxs)}/{r['n_chunks']} shown, non-contiguous)]")
        ax.set_title(
            f"{qtag}RPFI={r['RPFI']:.1f}   model={r['model']}   participant={r['participant']}   "
            f"{coverage}\n"
            f"[BWMD={r['comp']['BWMD']:.3f} SRE={r['comp']['SRE']:.3f} "
            f"WCR={r['comp']['WCR']:.3f} EDD={r['comp']['EDD']:.3f}]",
            fontsize=8, loc="left")
        if len(idxs) < r["n_chunks"]:
            print(f"    ⚠ {r['model']}/{r['participant']}: 전체 {r['n_chunks']}개 chunk 중 "
                  f"{len(idxs)}개를 고르게 퍼뜨려 표시(chunk index: {idxs}) — "
                  f"SRE/WCR/EDD는 여전히 전체 {r['n_chunks']}개 기준")
        ax.set_ylabel("z-score")
        ax.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.9)

    axs[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  저장: {out_path}")


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
    ap.add_argument("--out", default="gallery")
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--eval-seconds", type=float, default=10.0)
    ap.add_argument("--no-detrend", action="store_true")
    ap.add_argument("--no-bpf", action="store_true")
    ap.add_argument("--lambda-detrend", type=int, default=100)
    ap.add_argument("--quantiles", default="10,50,90")
    ap.add_argument("--n-chunks-show", type=int, default=3,
                    help="예시당 이어붙여 보여줄 chunk 수 (10초 단위)")
    ap.add_argument("--subject-n-models", type=int, default=2,
                    help="subject 모드에서 상/하위 몇 개씩 보여줄지")
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

    print(f"dataset={args.dataset}  fs={fs}Hz  models={models}")
    print("전체 (model × participant) 기록의 RPFI 계산 중...")
    records = compute_all_records(args.runs, label_full, models, fs, seg_len,
                                  do_detrend, do_bpf, args.lambda_detrend,
                                  anchors, weights, args.agg)
    print(f"  총 {len(records)}개 기록")
    rpfis = np.array([r["RPFI"] for r in records])
    print(f"  RPFI 범위: {rpfis.min():.1f} ~ {rpfis.max():.1f}  "
          f"(median={np.median(rpfis):.1f})")

    meta_rows = []

    # ── [1] quantile 갤러리 ──
    quantiles = [float(x) for x in args.quantiles.split(",")]
    q_examples = select_quantile_examples(records, quantiles)
    print(f"\n[1] quantile 갤러리 ({quantiles}):")
    for q, r in q_examples:
        print(f"    p{q:>3.0f}: RPFI={r['RPFI']:6.2f}  {r['model']:<16s} {r['participant']}")
        meta_rows.append(dict(gallery="quantile", quantile=q, dataset=args.dataset,
                              model=r["model"], participant=r["participant"],
                              RPFI=r["RPFI"], **r["comp"]))
    plot_gallery(q_examples, fs, seg_len,
                out_dir / f"{args.dataset}_quantile_gallery.png",
                args.n_chunks_show,
                title=f"{args.dataset.upper()} — representative reconstructions "
                      f"by RPFI quantile ({quantiles})")

    # ── [2] subject 갤러리 ──
    subj, s_examples = select_subject_examples(records, args.subject_n_models)
    if subj is not None:
        print(f"\n[2] subject 갤러리 — participant '{subj}' "
              f"(모델 간 RPFI 스프레드 최대):")
        for q, r in s_examples:
            print(f"    RPFI={r['RPFI']:6.2f}  {r['model']}")
            meta_rows.append(dict(gallery="subject", quantile="", dataset=args.dataset,
                                  model=r["model"], participant=r["participant"],
                                  RPFI=r["RPFI"], **r["comp"]))
        plot_gallery(s_examples, fs, seg_len,
                    out_dir / f"{args.dataset}_subject_gallery.png",
                    args.n_chunks_show,
                    title=f"{args.dataset.upper()} — participant '{subj}', "
                          f"reconstructions across models")
    else:
        print("\n[2] subject 갤러리: 조건을 만족하는 participant를 찾지 못해 건너뜀 "
              f"(모델 수가 최소 {max(3, 2*args.subject_n_models)}개 이상 필요)")

    # ── 메타데이터 저장 (캡션/재현용) ──
    meta_path = out_dir / f"{args.dataset}_gallery_meta.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta_rows[0].keys()))
        w.writeheader(); w.writerows(meta_rows)
    print(f"\n메타데이터: {meta_path}")


if __name__ == "__main__":
    main()

"""
cross_dataset_eval.py — Cross-dataset 일반화 검증 (재학습 없음, 기존 K-fold 체크포인트 재사용)
=====================================================================================
리뷰어 요구 (Major Comment 12):
  "A genuine cross-dataset experiment is needed because within-dataset training and
   testing cannot establish generalization."

지금까지 한 5-fold CV는 전부 "같은 데이터셋 안에서 subject만 나눈" 실험이었다.
이 스크립트는 완전히 다른 데이터셋에 대한 zero-shot 성능을 본다.

설계 — 왜 재학습이 필요 없는가
---------------------------------
이미 학습해둔 source 데이터셋의 K-fold 체크포인트(각 fold는 그 데이터셋 subject의
~80%로 학습됨, 나머지 20%는 outer test로 한 번도 안 씀)를 그대로 가져와서,
target 데이터셋 **전체**(단 한 명도 학습에 등장한 적 없음)에 바로 추론만 돌린다.
outer test subject가 원래도 "학습에 안 쓰인 데이터"였던 것처럼, target 데이터셋
전체도 이 모델 입장에선 완전히 새로운 데이터라 재학습 없이 정당한 zero-shot 평가가
성립한다.

K개 fold 체크포인트를 전부 쓰면 "K번 반복측정"이 되어, cross-dataset 성능의
불확실성 추정치도 추가 비용 없이 얻는다(fold별로 학습에 쓰인 subject 구성이 달라
약간씩 다른 모델이 나오므로).

출력은 train_models.py와 완전히 동일한 스키마(*_test_pred.npy + *_info.json)라
rpfi_eval.py를 그대로 --dataset {target} --runs runs/cross_{source}_to_{target} 로
돌리면 채점된다(수정 불필요).

⚠️ fs가 다른 데이터셋 간 이동 시 주의
----------------------------------------
UBFC/PURE는 fs=30(seq_len=300=10초), COHFACE는 fs=20(seq_len=200=10초)이다.
이 스크립트는 source 모델이 학습된 seq_len(샘플 개수)을 그대로 target 신호에
적용한다 — 즉 UBFC→COHFACE 방향이면 COHFACE 신호를 300샘플(=15초, 10초 아님)
단위로 잘라 추론한다. 초 단위를 맞추려면 리샘플링이 필요한데, 이번 실험은
"모델이 학습한 샘플 단위 패턴이 새 데이터셋에도 통하는가"를 보는 것이 목적이라
리샘플링 없이 진행한다. 이 사실은 반드시 논문 텍스트에 명시할 것.

사용법 (한 방향)
----------------
    python cross_dataset_eval.py \
        --source-dataset ubfc --source-runs runs/ubfc_cv5 \
        --target-dataset pure \
        --target-pred "../real data/PURE_POS_prediction.npy" \
        --target-label "../real data/PURE_label.npy" \
        --target-cv-split splits/pure_cv5.json \
        --out runs

6방향 전부 원하면 이 커맨드를 source/target 조합만 바꿔 6번 실행할 것
(아래 run_all_directions.sh 예시 참고).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

from models import build_model
from subject_split_v2 import load_cv, extract_by_ranges
from train_models import predict_test, eval_metrics
from rpfi_eval import discover_models, FS_BY_DATASET


def full_coverage(cv):
    """K-fold cv json에서 '전체 dataset'의 ranges/subject를 재구성한다.
    아무 fold나 train_ranges+test_ranges를 합치면 전체 subject를 커버한다는
    K-fold 정의상 성질을 이용 (diagnose_cohface.py/rpfi_gallery.py와 동일 트릭)."""
    f0 = cv["folds"][0]
    ranges = list(f0["train_ranges"]) + list(f0["test_ranges"])
    subj_of = list(f0["train_subject_of_range"]) + list(f0["test_subject_of_range"])
    return ranges, subj_of


def resolve_ckpt(raw_path, src_runs):
    """info.json에 저장된 체크포인트 경로. 실행 당시 cwd가 지금과 다르면 안 열릴 수
    있으니, 안 되면 src_runs 안의 같은 파일명으로 한 번 더 시도한다."""
    ck = Path(raw_path)
    if ck.exists():
        return ck
    ck2 = src_runs / ck.name
    if ck2.exists():
        return ck2
    raise SystemExit(f"체크포인트를 찾을 수 없습니다: {raw_path} (시도한 대체 경로: {ck2})")


def process_blend_model(model_name, src_runs, args, t_x, t_y, t_ranges, t_subj_of, device, out_dir):
    """bimamba_blend는 fold당 단일 .pth가 아니라 inner 모델 K_inner개 + blend_weights로
    구성돼있다(train_bimamba_blend.py 참고). info.json의 inner_models[i]['ckpt']를 각각
    불러와 target에 추론한 뒤, 저장된 blend_weights로 가중합한다.
    ※ info.json에 seq_len이 저장돼있지 않아(v2 설계 당시 누락), source 데이터셋의
      표준 관례(FS_BY_DATASET[source]*10초)로 역산한다 — 다른 모델들과 동일한 값이다."""
    seq_len = FS_BY_DATASET[args.source_dataset] * 10
    info_files = sorted(src_runs.glob(f"{model_name}_f*_info.json"),
                        key=lambda p: int(re.search(r"_f(\d+)_info\.json$", p.name).group(1)))
    results = []
    for info_p in info_files:
        info = json.load(open(info_p))
        fold = info["fold"]
        inner_models = info["inner_models"]
        weights = np.asarray(info["blend_weights"], dtype=float)
        if len(inner_models) != len(weights):
            raise SystemExit(f"{info_p}: inner_models({len(inner_models)})과 "
                             f"blend_weights({len(weights)}) 개수 불일치")

        preds_per_inner = []
        for im in inner_models:
            ck = resolve_ckpt(im["ckpt"], src_runs)
            m = build_model("bimamba").to(device)   # blend의 inner는 항상 bimamba 아키텍처
            m.load_state_dict(torch.load(ck, map_location=device))
            preds_per_inner.append(predict_test(m, t_x, seq_len, device, batch=args.batch))

        final_preds = []
        for rec_i in range(len(t_x)):
            acc = np.zeros_like(preds_per_inner[0][rec_i])
            for w, mp in zip(weights, preds_per_inner):
                acc = acc + w * mp[rec_i]
            final_preds.append(acc.astype(np.float32))

        metrics = eval_metrics(t_y, final_preds)
        print(f"  [{model_name} src_f{fold}] seq_len={seq_len} (K_inner={len(inner_models)}, "
              f"역산값 — info.json에 미저장)  Pearson={metrics['pearson_mean']:+.4f}  "
              f"RMSE={metrics['rmse']:.4f}  n_rec={metrics['n_recordings']}")

        np.save(out_dir / f"{model_name}_f{fold}_test_pred.npy",
                np.array(final_preds, dtype=object), allow_pickle=True)
        out_info = dict(
            model=model_name, fold=fold,
            source_dataset=args.source_dataset, target_dataset=args.target_dataset,
            source_ckpts=[str(resolve_ckpt(im["ckpt"], src_runs)) for im in inner_models],
            blend_weights=info["blend_weights"], seq_len=int(seq_len),
            test_ranges=[list(r) for r in t_ranges],
            test_subject_of_range=t_subj_of,
            test_rec_lengths=[int(len(p)) for p in final_preds],
            pred_space="window_znorm",
            test_metrics=metrics,
        )
        json.dump(out_info, open(out_dir / f"{model_name}_f{fold}_info.json", "w"),
                  indent=2, default=str)
        results.append(out_info)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--source-runs", required=True, help="이미 학습된 K-fold 결과 디렉토리")
    ap.add_argument("--target-dataset", required=True, choices=["pure", "ubfc", "cohface"])
    ap.add_argument("--target-pred", required=True)
    ap.add_argument("--target-label", required=True)
    ap.add_argument("--target-cv-split", required=True,
                    help="target의 cv json (K-fold 구조는 안 쓰고 전체 subject 목록/ranges만 재사용)")
    ap.add_argument("--models", default=None, help="콤마 구분. 미지정시 source-runs에서 자동 탐색")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    if args.source_dataset == args.target_dataset:
        raise SystemExit("source와 target이 같습니다 — cross-dataset 실험이 아닙니다. "
                         "같은 데이터셋 내 일반화는 이미 K-fold CV로 확인했습니다.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_pred = np.load(args.target_pred)
    target_label = np.load(args.target_label)
    target_cv = load_cv(args.target_cv_split)
    t_ranges, t_subj_of = full_coverage(target_cv)
    t_x = extract_by_ranges(target_pred, t_ranges)
    t_y = extract_by_ranges(target_label, t_ranges)
    n_target_subjects = len(set(t_subj_of))

    src_runs = Path(args.source_runs)
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else discover_models(src_runs))
    if not models:
        raise SystemExit(f"{src_runs} 에서 모델을 찾지 못했습니다.")

    out_dir = Path(args.out) / f"cross_{args.source_dataset}_to_{args.target_dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"cross-dataset: {args.source_dataset} → {args.target_dataset}  "
          f"target subjects={n_target_subjects}명 (전부 미학습, 완전 zero-shot)  device={device}")
    print(f"모델 {len(models)}개: {models}")

    summary = []
    for model_name in models:
        if model_name == "bimamba_blend":
            summary.extend(process_blend_model(model_name, src_runs, args,
                                               t_x, t_y, t_ranges, t_subj_of,
                                               device, out_dir))
            continue

        # "{model}_f{digits}.pth"에 정확히 일치하는 것만 (bimamba_blend의 "_f0_inner0.pth"
        # 같은 파일이 다른 모델 이름에 우연히 걸리는 것도 방지)
        pat = re.compile(rf"^{re.escape(model_name)}_f(\d+)$")
        ckpts = []
        for p in src_runs.glob(f"{model_name}_f*.pth"):
            m = pat.match(p.stem)
            if m:
                ckpts.append((int(m.group(1)), p))
        ckpts.sort(key=lambda x: x[0])
        if not ckpts:
            print(f"  [건너뜀] {model_name}: 체크포인트 없음 ({src_runs}/{model_name}_f<N>.pth)")
            continue

        for fold, ck in ckpts:
            info_p = src_runs / f"{model_name}_f{fold}_info.json"
            if not info_p.exists():
                print(f"  [건너뜀] {ck.name}: 대응하는 info.json 없음")
                continue
            src_info = json.load(open(info_p))
            seq_len = src_info.get("seq_len")
            if seq_len is None:
                raise SystemExit(f"{info_p} 에 seq_len이 없습니다 — train_models.py 산출물이 맞는지 확인하세요.")

            model = build_model(model_name).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))

            preds = predict_test(model, t_x, seq_len, device, batch=args.batch)
            metrics = eval_metrics(t_y, preds)

            print(f"  [{model_name} src_f{fold}] seq_len={seq_len}  "
                  f"Pearson={metrics['pearson_mean']:+.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"n_rec={metrics['n_recordings']}")

            np.save(out_dir / f"{model_name}_f{fold}_test_pred.npy",
                    np.array(preds, dtype=object), allow_pickle=True)
            info = dict(
                model=model_name, fold=fold,
                source_dataset=args.source_dataset, target_dataset=args.target_dataset,
                source_ckpt=str(ck), seq_len=int(seq_len),
                test_ranges=[list(r) for r in t_ranges],
                test_subject_of_range=t_subj_of,
                test_rec_lengths=[int(len(p)) for p in preds],
                pred_space="window_znorm",
                test_metrics=metrics,
            )
            json.dump(info, open(out_dir / f"{model_name}_f{fold}_info.json", "w"),
                      indent=2, default=str)
            summary.append(info)

    if not summary:
        raise SystemExit("생성된 결과가 없습니다 — 체크포인트 경로를 확인하세요.")

    by_model = {}
    for s in summary:
        by_model.setdefault(s["model"], []).append(s["test_metrics"]["pearson_mean"])
    print(f"\n=== {args.source_dataset} → {args.target_dataset} 모델별 집계 (source fold 반복 평균) ===")
    for m, ps in sorted(by_model.items(), key=lambda x: -np.nanmean(x[1])):
        ps = np.array(ps, dtype=float)
        print(f"  {m:<16s} n_fold={len(ps)}  Pearson {np.nanmean(ps):+.4f}±{np.nanstd(ps):.4f}")

    sp = out_dir / "summary.json"
    json.dump(summary, open(sp, "w"), indent=2, default=str)
    print(f"\n완료 -> {out_dir.resolve()}")
    print(f"채점(RPFI): python rpfi_eval.py --dataset {args.target_dataset} "
          f"--runs {out_dir} --label \"{args.target_label}\" "
          f"--anchors anchors/rpfi_anchors.json "
          f"--out results_cross_{args.source_dataset}_to_{args.target_dataset}")


if __name__ == "__main__":
    main()

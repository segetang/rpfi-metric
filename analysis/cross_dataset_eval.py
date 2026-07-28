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

★ fs가 다른 데이터셋 간 이동 시 리샘플링 보정 (v2)
----------------------------------------------------
UBFC/PURE는 fs=30(seq_len=300=10초), COHFACE는 fs=20(seq_len=200=10초)이다.
모델은 seq_len을 "샘플 개수"로 고정해 학습되므로, source seq_len을 target 신호에
그대로(리샘플링 없이) 적용하면 물리적 윈도우 길이가 어긋난다 — 예를 들어
COHFACE(fs=20)로 학습한 모델을 UBFC(fs=30)에 적용할 때 seq_len=200을 그대로 쓰면
200/30=6.67초 분량만 보게 되고, 반대로 UBFC/PURE(fs=30)로 학습한 모델을
COHFACE(fs=20)에 적용하면 300/20=15초 분량을 보게 된다 — 둘 다 모델이 학습 때
경험한 "10초"라는 물리적 컨텍스트를 벗어난다.

v1(최초 버전)은 이 사실을 몰랐던 게 아니라 "리샘플링 없이 원시 샘플 패턴이
새 데이터셋에도 통하는지를 본다"는 의도적 설계였지만, 이후 진단
(a_crossdirection_volatility.csv, e_structural_mismatch.csv, d_signaldiag_*.csv)에서
이 윈도우 불일치가 실제로 physdiff 같은 고정길이 의존 아키텍처의 cross-dataset
변동성을 지배적으로 설명한다는 게 드러나 — "일반화 실패"와 "윈도우 아티팩트"가
뒤섞여 해석 불가능해지는 문제가 있었다. v2는 이를 제거한다:

  1. target 신호(t_x)를 추론 직전에 target_fs → source_fs로 유리수비
     리샘플링(scipy.signal.resample_poly)해, 모델이 항상 물리적으로 정확히
     10초(=source_fs*10 샘플)를 보게 만든다.
  2. 모델 출력(예측)을 다시 source_fs → target_fs로 리샘플링해 되돌린다 —
     target_label(t_y)과 길이/샘플링이 맞아야 rpfi_eval.py가 target 데이터셋
     고유의 chunk 구조로 그대로 채점할 수 있기 때문이다.
  3. 리샘플링 반올림 오차(최대 몇 샘플)는 t_y 길이에 맞춰 자르거나 마지막 값을
     이어붙여 보정한다(match_length).

같은 fs끼리(pure↔ubfc)는 리샘플링이 항등변환이라 v1과 결과가 동일하다 —
재실행이 필요한 건 COHFACE가 source 또는 target인 4방향뿐이다.

--no-resample 플래그로 v1(구버전, 원시 샘플 그대로) 동작을 재현할 수 있다 —
보정 전/후 영향을 정량적으로 보여주고 싶을 때 ablation 용도로 사용.

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
(아래 run_all_directions.sh 예시 참고). fs가 같은 pure↔ubfc 2방향은 v1 결과를
그대로 써도 무방하다(리샘플링이 no-op이므로 재실행해도 수치는 동일).
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from models import build_model
from subject_split_v2 import load_cv, extract_by_ranges
from train_models import predict_test, eval_metrics
from rpfi_eval import discover_models, FS_BY_DATASET


# ══════════════════════════════════════════════════════════════════
# fs 불일치 보정 — 물리적 윈도우 길이 보존
# ══════════════════════════════════════════════════════════════════

def resample_signal(x, fs_from, fs_to):
    """1D 신호를 fs_from → fs_to 로 유리수비(rational) 리샘플링한다.
    물리적 지속시간(초)을 보존하는 것이 목적이며, 표본 개수는 fs 비율에 맞춰 바뀐다.
    fs_from == fs_to 이거나 빈 배열이면 그대로 반환(no-op)."""
    x = np.asarray(x, dtype=np.float64)
    if fs_from == fs_to or len(x) == 0:
        return x.astype(np.float32)
    g = math.gcd(int(fs_from), int(fs_to))
    up, down = int(fs_to // g), int(fs_from // g)
    y = resample_poly(x, up, down)
    return y.astype(np.float32)


def resample_recordings(x_list, fs_from, fs_to):
    """recording별 1D 배열 리스트 전체에 resample_signal을 적용."""
    return [resample_signal(x, fs_from, fs_to) for x in x_list]


def match_length(x, target_len):
    """리샘플링 반올림 오차(최대 몇 샘플) 보정: target_len에 맞춰 자르거나
    마지막 값을 이어붙여 채운다(zero-pad 대신 edge-pad — 생리 신호에 인위적
    불연속을 만들지 않기 위함)."""
    n = len(x)
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    pad_val = x[-1] if n > 0 else np.float32(0.0)
    pad = np.full(target_len - n, pad_val, dtype=x.dtype)
    return np.concatenate([x, pad])


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


def _postprocess_preds(preds_raw, t_y, fs_source, fs_target, resample_needed):
    """모델 출력(source_fs)을 target_fs로 되돌리고 t_y 길이에 맞춘다.
    resample_needed=False면 원본 그대로(v1과 동일 동작)."""
    if not resample_needed:
        return preds_raw
    return [match_length(resample_signal(p, fs_source, fs_target), len(y))
            for p, y in zip(preds_raw, t_y)]


def process_blend_model(model_name, src_runs, args, t_x_model_input, t_y, t_ranges, t_subj_of,
                         device, out_dir, fs_source, fs_target, resample_needed):
    """bimamba_blend는 fold당 단일 .pth가 아니라 inner 모델 K_inner개 + blend_weights로
    구성돼있다(train_bimamba_blend.py 참고). info.json의 inner_models[i]['ckpt']를 각각
    불러와 target에 추론한 뒤, 저장된 blend_weights로 가중합한다.
    ※ info.json에 seq_len이 저장돼있지 않아(v2 설계 당시 누락), source 데이터셋의
      표준 관례(FS_BY_DATASET[source]*10초)로 역산한다 — 다른 모델들과 동일한 값이다.
    ※ 리샘플링은 inner 모델별로 반복하지 않고, blend 이후 최종 예측 1회에만 적용한다
      (inner 모델 전부 동일 아키텍처/seq_len이라 입력 쪽은 이미 t_x_model_input로 통일됨)."""
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
            preds_per_inner.append(predict_test(m, t_x_model_input, seq_len, device, batch=args.batch))

        final_preds_raw = []
        for rec_i in range(len(t_x_model_input)):
            acc = np.zeros_like(preds_per_inner[0][rec_i])
            for w, mp in zip(weights, preds_per_inner):
                acc = acc + w * mp[rec_i]
            final_preds_raw.append(acc.astype(np.float32))

        final_preds = _postprocess_preds(final_preds_raw, t_y, fs_source, fs_target, resample_needed)

        metrics = eval_metrics(t_y, final_preds)
        resample_tag = f"resampled({fs_source}→{fs_target}Hz)" if resample_needed else "no-resample"
        print(f"  [{model_name} src_f{fold}] seq_len={seq_len} (K_inner={len(inner_models)}, "
              f"역산값 — info.json에 미저장) {resample_tag}  Pearson={metrics['pearson_mean']:+.4f}  "
              f"RMSE={metrics['rmse']:.4f}  n_rec={metrics['n_recordings']}")

        np.save(out_dir / f"{model_name}_f{fold}_test_pred.npy",
                np.array(final_preds, dtype=object), allow_pickle=True)
        out_info = dict(
            model=model_name, fold=fold,
            source_dataset=args.source_dataset, target_dataset=args.target_dataset,
            source_ckpts=[str(resolve_ckpt(im["ckpt"], src_runs)) for im in inner_models],
            blend_weights=info["blend_weights"], seq_len=int(seq_len),
            fs_source=int(fs_source), fs_target=int(fs_target),
            resampled=bool(resample_needed), window_seconds=10.0,
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
    ap.add_argument("--no-resample", action="store_true",
                    help="fs가 달라도 리샘플링하지 않고 v1(구버전) 방식대로 원시 샘플 수 그대로 "
                         "windowing한다 — 보정 전/후 비교용 ablation 옵션. 기본값은 리샘플링 적용(권장).")
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

    fs_source = FS_BY_DATASET[args.source_dataset]
    fs_target = FS_BY_DATASET[args.target_dataset]
    resample_needed = (fs_source != fs_target) and not args.no_resample

    if fs_source != fs_target and args.no_resample:
        print(f"  ⚠ --no-resample 지정됨: fs 불일치({fs_source}Hz→{fs_target}Hz)를 보정하지 않고 "
              f"v1 방식(원시 샘플 수 그대로)으로 진행합니다 — 물리적 윈도우 길이가 "
              f"{fs_source * 10 / fs_target:.2f}초로 어긋난 채 평가됩니다(ablation 목적일 때만 사용).")
        t_x_model_input = t_x
    elif resample_needed:
        print(f"  ℹ fs 불일치 감지({args.source_dataset}={fs_source}Hz vs "
              f"{args.target_dataset}={fs_target}Hz) — 모델 입력을 {fs_source}Hz로 리샘플링해 "
              f"물리적 10초 윈도우를 보존하고, 예측을 다시 {fs_target}Hz로 되돌려 채점합니다.")
        t_x_model_input = resample_recordings(t_x, fs_target, fs_source)
    else:
        t_x_model_input = t_x  # fs 동일 → no-op, v1과 완전히 동일한 결과

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
                                               t_x_model_input, t_y, t_ranges, t_subj_of,
                                               device, out_dir, fs_source, fs_target, resample_needed))
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

            preds_raw = predict_test(model, t_x_model_input, seq_len, device, batch=args.batch)
            preds = _postprocess_preds(preds_raw, t_y, fs_source, fs_target, resample_needed)
            metrics = eval_metrics(t_y, preds)

            resample_tag = f"resampled({fs_source}→{fs_target}Hz)" if resample_needed else "no-resample"
            print(f"  [{model_name} src_f{fold}] seq_len={seq_len} {resample_tag}  "
                  f"Pearson={metrics['pearson_mean']:+.4f}  RMSE={metrics['rmse']:.4f}  "
                  f"n_rec={metrics['n_recordings']}")

            np.save(out_dir / f"{model_name}_f{fold}_test_pred.npy",
                    np.array(preds, dtype=object), allow_pickle=True)
            info = dict(
                model=model_name, fold=fold,
                source_dataset=args.source_dataset, target_dataset=args.target_dataset,
                source_ckpt=str(ck), seq_len=int(seq_len),
                fs_source=int(fs_source), fs_target=int(fs_target),
                resampled=bool(resample_needed), window_seconds=10.0,
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

"""
investigate_physdiff_asymmetry.py — physdiff cross-dataset 비대칭 원인 조사
=============================================================================
§4-h에서 새로 발견된 현상:
  PURE   → UBFC : physdiff RPFI 1위 (32.14, 8개 중 최고)
  COHFACE→ UBFC : physdiff RPFI 꼴찌 (15.17, 8개 중 최저)
같은 physdiff가 어느 source로 학습했느냐에 따라 정반대 극단을 오간다.
논문에 "관찰됨"으로만 적을지, 원인까지 파고들지 결정하기 위한 진단 스크립트.

5단계로 원인을 좁혀 들어간다:
  (A) 이게 정말 physdiff만의 특이 현상인가?
      -> 전 모델의 cross-direction RPFI 변동폭을 비교해 physdiff의 순위를 확인
  (B) 어느 컴포넌트(BWMD/SRE/WCR/EDD)가 붕괴를 주도하는가?
      -> 이미 계산된 rpfi_participant_*.csv에서 physdiff 행만 비교
  (C) "COHFACE로 학습된 physdiff 자체가 이미 퇴화된 모델"인가?
      -> 학습 시점 info.json(best_val/epochs_run/convergence)을
         COHFACE에서 학습된 다른 모델(대조군)과 비교
  (D) "예측 신호 자체가 이상한가" (진폭 붕괴 / 대역 이탈 / 비주기적)?
      -> 원시 예측 npy의 스펙트럼·표준편차 직접 진단
  (E) "seq_len 불일치로 인한 시간축 스케일 미스매치"인가? (§5 fs 주의사항과 동일 이슈)
      -> cross_dataset_eval.py는 source가 학습한 seq_len을 target에 그대로 적용한다.
         COHFACE(fs=20, seq_len=200)로 학습한 모델을 UBFC(fs=30)에 적용하면
         "200샘플" 윈도우가 실제로는 6.67초(=200/30)로, 학습 시 보았던 10초(=200/20)
         윈도우보다 훨씬 짧다 -> 윈도우당 심박 주기 개수가 달라진다.
         이 계산은 데이터 없이도 바로 가능하므로 가장 먼저 확인할 가치가 있다.
         (단, COHFACE→PURE도 같은 seq_len 불일치를 겪지만 physdiff가 거기선
          꼴찌가 아니라 2위였다 — 그래서 "필요조건일 수는 있어도 충분조건은
          아니다"라는 게 이 스크립트의 실측이 답해야 할 질문이다.)

전제 (프로젝트 컨벤션 기반 추정 — 실제와 다르면 CONFIG 섹션만 고치면 됨):
  - runs/{dataset}_cv5/{model}_f{k}_info.json           (train_models.py 산출물, 스키마 확인됨)
  - runs/{dataset}_cv5/{model}_f{k}_test_pred.npy        (object array, recording별 1D, window-znorm)
  - runs/cross_{source}_to_{target}/{model}_f{k}_test_pred.npy   (실측 확인됨, --runs-root 기준)
  - results_cross_{source}_to_{target}/rpfi_participant_{target}_{agg}.csv (§4-h에서 언급된 경로, --results-root 기준)
  info.json 스키마는 train_models.py에서 직접 확인한 필드만 사용:
    seq_len, test_subject_of_range, test_rec_lengths, convergence{status}, best_val, epochs_run

사용 예:
    python investigate_physdiff_asymmetry.py \
        --results-root "../real data/.." --runs-root runs \
        --target ubfc --sources pure,cohface --model physdiff --contrast-model rnn
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

CARDIAC_BAND = (0.75, 3.0)          # EDD/SRE와 동일 대역
FS = {"pure": 30, "ubfc": 30, "cohface": 20}
NATIVE_SEQ_LEN = {"pure": 300, "ubfc": 300, "cohface": 200}  # chunk-mode=seconds, 10초 기준


# ══════════════════════════════════════════════════════════════
# 공용 유틸
# ══════════════════════════════════════════════════════════════

def find_col(df, keywords):
    """컬럼명을 자동 탐지. 못 찾으면 None.
    정확히 일치하는 컬럼(대소문자 무시)을 최우선으로 보고, 그다음 부분일치.
    ★ 부분일치만 쓰면 'RPFI'를 찾을 때 'RPFI_alt_agg'가 먼저 걸릴 수 있어
      (둘 다 'rpfi'를 포함) — 반드시 정확일치를 먼저 시도해야 함."""
    for kw in keywords:
        for c in df.columns:
            if c.lower() == kw.lower():
                return c
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None


def glob_one(root, pattern, what):
    cands = sorted(Path(root).glob(pattern))
    if not cands:
        existing = [p.name for p in Path(root).glob("*")] if Path(root).exists() else ["<폴더 자체가 없음>"]
        raise FileNotFoundError(
            f"[{what}] {root}/{pattern} 를 못 찾았습니다.\n"
            f"  실제 폴더 내용: {existing[:20]}\n"
            f"  -> 파일명 컨벤션이 다르면 --*-glob 옵션으로 직접 지정하세요.")
    return cands[0]


# ══════════════════════════════════════════════════════════════
# (A) 전 모델 cross-direction 변동폭
# ══════════════════════════════════════════════════════════════

def load_cross_rpfi_csv(results_root, source, target, agg):
    d = Path(results_root) / f"results_cross_{source}_to_{target}"
    pat = f"rpfi_participant_{target}_{agg}*.csv"
    try:
        f = glob_one(d, pat, f"RPFI CSV ({source}->{target})")
    except FileNotFoundError:
        f = glob_one(d, "rpfi_participant_*.csv", f"RPFI CSV ({source}->{target})")
    return pd.read_csv(f), f


def part_a_volatility(df_a, df_b, model_col, rpfi_col, label_a, label_b):
    ga = df_a.groupby(model_col)[rpfi_col].mean()
    gb = df_b.groupby(model_col)[rpfi_col].mean()
    common = sorted(set(ga.index) & set(gb.index))
    out = pd.DataFrame({label_a: ga[common], label_b: gb[common]})
    out["abs_diff"] = (out[label_a] - out[label_b]).abs()
    out["swing_rank(1=최대변동)"] = out["abs_diff"].rank(ascending=False).astype(int)
    return out.sort_values("abs_diff", ascending=False)


# ══════════════════════════════════════════════════════════════
# (B) 컴포넌트 분해
# ══════════════════════════════════════════════════════════════

def part_b_components(df_a, df_b, model_col, model_name, label_a, label_b):
    comp_keywords = {"BWMD": ["bwmd"], "SRE": ["sre"], "WCR": ["wcr"], "EDD": ["edd"],
                      "HR_MAE": ["hr_mae", "hr mae"]}
    rows = []
    for label, df in [(label_a, df_a), (label_b, df_b)]:
        sub = df[df[model_col].astype(str).str.lower() == model_name.lower()]
        if sub.empty:
            warnings.warn(f"{label}: 모델 '{model_name}' 행을 못 찾음 (실제 값: "
                          f"{df[model_col].unique()[:10]})")
            continue
        row = {"direction": label, "n_participants": len(sub)}
        for comp, kws in comp_keywords.items():
            col = find_col(sub, kws)
            if col:
                row[f"{comp}_mean"] = round(sub[col].mean(), 4)
                row[f"{comp}_std"] = round(sub[col].std(), 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# (C) 학습 진단 (info.json)
# ══════════════════════════════════════════════════════════════

def part_c_train_diag(runs_root, dataset, model_name, n_folds=5, info_glob=None):
    root = Path(runs_root) / f"{dataset}_cv{n_folds}"
    pattern = info_glob or f"{model_name}_f{{k}}_info.json"
    rows = []
    for k in range(n_folds):
        f = root / pattern.format(k=k)
        if not f.exists():
            warnings.warn(f"info.json 없음: {f}")
            continue
        info = json.loads(f.read_text())
        conv = info.get("convergence", {})
        rows.append({
            "fold": k,
            "best_val": info.get("best_val"),
            "epochs_run": info.get("epochs_run"),
            "status": conv.get("status") if isinstance(conv, dict) else conv,
            "tail_rel_gain": conv.get("tail_rel_gain") if isinstance(conv, dict) else None,
            "test_pearson": (info.get("test_metrics") or {}).get("pearson_mean"),
            "seq_len": info.get("seq_len"),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# (D) 원시 예측 신호 스펙트럼/주기성 진단
# ══════════════════════════════════════════════════════════════

def part_d_signal_diag(runs_root, source, target, model_name, fs_target,
                       n_folds=5, pred_glob=None, max_rec_per_fold=None):
    """★ 실측 확인됨(사용자 find 결과): 예측 npy는 results_cross_*/가 아니라
    runs/cross_{source}_to_{target}/{model}_f{k}_test_pred.npy 에 있다.
    즉 --results-root가 아니라 --runs-root 기준."""
    root = Path(runs_root) / f"cross_{source}_to_{target}"
    pattern = pred_glob or f"{model_name}_f{{k}}_test_pred.npy"
    rows = []
    for k in range(n_folds):
        f = root / pattern.format(k=k)
        if not f.exists():
            warnings.warn(f"예측 파일 없음: {f}")
            continue
        preds = np.load(f, allow_pickle=True)
        n_rec = len(preds) if max_rec_per_fold is None else min(max_rec_per_fold, len(preds))
        for r in range(n_rec):
            x = np.asarray(preds[r], dtype=np.float64)
            if x.size < fs_target * 2:
                continue
            f_w, pxx = sps.welch(x, fs=fs_target, nperseg=min(256, len(x)))
            band = (f_w >= CARDIAC_BAND[0]) & (f_w <= CARDIAC_BAND[1])
            total = pxx.sum() + 1e-12
            band_power = pxx[band].sum() if band.any() else 0.0
            dom_freq = f_w[band][np.argmax(pxx[band])] if band.any() and pxx[band].sum() > 0 else np.nan
            rows.append({
                "fold": k, "recording": r, "n_samples": x.size,
                "std": x.std(), "dom_freq_hz": dom_freq,
                "cardiac_band_power_ratio": band_power / total,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# (E) 구조적 seq_len/fs 미스매치 정량화 — 데이터 없이 바로 계산 가능
# ══════════════════════════════════════════════════════════════

def part_e_structural_mismatch(sources, target, typical_hr_bpm=(55, 90)):
    """source에서 학습한 seq_len을 target(fs 다를 수 있음)에 그대로 적용했을 때
    윈도우의 '실제 시간 길이'와 '윈도우당 예상 심박 주기 개수'가 얼마나 어긋나는지 계산.
    (cross_dataset_eval.py가 source seq_len을 그대로 쓴다는 §5 기록 기반)
    """
    rows = []
    for src in sources + [target]:  # target도 자기 자신 기준(=미스매치 0)으로 같이 표기
        seq_len = NATIVE_SEQ_LEN[src]
        trained_seconds = seq_len / FS[src]
        applied_seconds = seq_len / FS[target]
        lo_cyc = applied_seconds * typical_hr_bpm[0] / 60
        hi_cyc = applied_seconds * typical_hr_bpm[1] / 60
        lo_cyc_tr = trained_seconds * typical_hr_bpm[0] / 60
        hi_cyc_tr = trained_seconds * typical_hr_bpm[1] / 60
        rows.append({
            "source": src, "target": target,
            "seq_len(고정)": seq_len,
            "fs_source": FS[src], "fs_target": FS[target],
            "학습시_윈도우_초": round(trained_seconds, 2),
            f"{target}적용시_윈도우_초": round(applied_seconds, 2),
            "시간축_배율(적용/학습)": round(applied_seconds / trained_seconds, 3),
            "학습시_예상_심박수/윈도우": f"{lo_cyc_tr:.1f}~{hi_cyc_tr:.1f}",
            f"{target}적용시_예상_심박수/윈도우": f"{lo_cyc:.1f}~{hi_cyc:.1f}",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", default=".",
                    help="results_cross_{source}_to_{target}/ 들의 상위 경로")
    ap.add_argument("--runs-root", default="runs",
                    help="{dataset}_cv5/ 들의 상위 경로 (within-dataset 학습 산출물)")
    ap.add_argument("--target", default="ubfc")
    ap.add_argument("--sources", default="pure,cohface",
                    help="비교할 두 source, 콤마 구분 (첫번째=강세/두번째=약세 방향이라고 가정하지 않음)")
    ap.add_argument("--model", default="physdiff")
    ap.add_argument("--contrast-model", default="rnn",
                    help="COHFACE→UBFC에서 physdiff 대신 1위였던 모델 (대조군)")
    ap.add_argument("--fs-target", type=float, default=None,
                    help="미지정시 FS 딕셔너리에서 --target 기준으로 자동 설정")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--agg", default="geometric")
    ap.add_argument("--pred-glob", default=None,
                    help="cross 예측 npy 파일명 패턴 (기본 '{model}_f{k}_test_pred.npy'). "
                         "실제 파일명이 다르면 여기서 맞추세요 (예: '{model}_f{k}_pred.npy')")
    ap.add_argument("--info-glob", default=None,
                    help="within-dataset info.json 파일명 패턴 (기본 '{model}_f{k}_info.json')")
    ap.add_argument("--skip-signal-diag", action="store_true",
                    help="원시 npy 없이 (A)(B)(C)(E)만 실행 (CSV/JSON만 있어도 됨)")
    ap.add_argument("--out", default="physdiff_asymmetry_report")
    args = ap.parse_args()

    fs_target = args.fs_target or FS.get(args.target)
    if fs_target is None:
        raise SystemExit(f"--target '{args.target}'의 fs를 모릅니다. --fs-target으로 직접 지정하세요.")

    sources = [s.strip() for s in args.sources.split(",")]
    assert len(sources) == 2, "--sources는 정확히 2개여야 합니다 (예: pure,cohface)"
    src_a, src_b = sources
    label_a, label_b = f"{src_a}→{args.target}", f"{src_b}→{args.target}"

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    # (E) 구조적 미스매치는 데이터 없이 바로 계산 — 항상 먼저 출력
    print("=" * 70)
    print("(E) 구조적 seq_len/fs 미스매치 (데이터 불필요, 항상 계산 가능)")
    print("=" * 70)
    e = part_e_structural_mismatch(sources, args.target)
    print(e.to_string(index=False))
    e.to_csv(out_dir / "e_structural_mismatch.csv", index=False)
    print("-> '시간축_배율'이 1.0에서 먼 source일수록 윈도우 길이가 학습 때와 달라진다.")
    print("   COHFACE→PURE도 같은 배율을 겪는데 거기선 physdiff가 꼴찌가 아니었으므로,")
    print("   이 미스매치가 '필요조건일 수는 있으나 충분조건은 아님'을 염두에 두고 (A)~(D)를 볼 것.\n")

    # (A)
    print("=" * 70)
    print(f"(A) 전 모델 cross-direction RPFI 변동폭: {label_a} vs {label_b}")
    print("=" * 70)
    try:
        df_a, path_a = load_cross_rpfi_csv(args.results_root, src_a, args.target, args.agg)
        df_b, path_b = load_cross_rpfi_csv(args.results_root, src_b, args.target, args.agg)
        print(f"[로드] {path_a}\n[로드] {path_b}")
        model_col = find_col(df_a, ["model"])
        rpfi_col = find_col(df_a, ["rpfi"])   # 정확일치 우선이라 'RPFI'(geometric)가 잡힘, 'RPFI_alt_agg' 아님
        if model_col is None or rpfi_col is None:
            raise RuntimeError(f"model/RPFI 컬럼을 못 찾음. 실제 컬럼: {list(df_a.columns)}")
        print(f"[컬럼 감지] model='{model_col}', RPFI='{rpfi_col}'")
        vol = part_a_volatility(df_a, df_b, model_col, rpfi_col, label_a, label_b)
        print(vol.to_string())
        vol.to_csv(out_dir / "a_crossdirection_volatility.csv")
        physdiff_rank = vol.loc[vol.index.str.lower() == args.model.lower(),
                                "swing_rank(1=최대변동)"]
        if len(physdiff_rank):
            n = len(vol)
            print(f"\n-> {args.model}의 변동폭 순위: {int(physdiff_rank.iloc[0])}/{n} "
                  f"(1위=전 모델 중 가장 많이 흔들림)")

        print(f"\n=== (B) {args.model} 컴포넌트별 붕괴 지점 ===")
        comp = part_b_components(df_a, df_b, model_col, args.model, label_a, label_b)
        print(comp.to_string(index=False))
        comp.to_csv(out_dir / "b_component_breakdown.csv", index=False)
    except FileNotFoundError as e_:
        print(f"[건너뜀] {e_}")

    # (C)
    print("\n" + "=" * 70)
    print(f"(C) 학습 진단: {args.model}이 {src_a}/{src_b}에서 어떻게 학습됐나 (+ 대조군)")
    print("=" * 70)
    for src in (src_a, src_b):
        d = part_c_train_diag(args.runs_root, src, args.model, args.n_folds, args.info_glob)
        print(f"--- {args.model} trained on {src} ---")
        print(d.to_string(index=False) if len(d) else "  (없음)")
        d.to_csv(out_dir / f"c_traindiag_{args.model}_{src}.csv", index=False)
    d_contrast = part_c_train_diag(args.runs_root, src_b, args.contrast_model,
                                   args.n_folds, args.info_glob)
    print(f"--- (대조군) {args.contrast_model} trained on {src_b} ---")
    print(d_contrast.to_string(index=False) if len(d_contrast) else "  (없음)")
    d_contrast.to_csv(out_dir / f"c_traindiag_{args.contrast_model}_{src_b}.csv", index=False)
    print(f"\n-> {args.model}의 best_val/epochs_run/status가 대조군과 비슷한데 cross-transfer만")
    print(f"   유독 나쁘면: 학습 자체보다 아키텍처 특성(예: physdiff는 positional encoding이")
    print(f"   없음 — models.py 확인됨, disentangle의 1차 차분 기반 trend/amp만 사용)이")
    print(f"   원인일 가능성. 반대로 {src_b} 학습 시 status가 유독 나쁘면 '애초에 덜 학습된")
    print(f"   모델을 UBFC에 적용한 것뿐'이라는 더 단순한 설명이 우선됨.")

    # (D)
    if not args.skip_signal_diag:
        print("\n" + "=" * 70)
        print("(D) 예측 신호 스펙트럼/주기성 진단")
        print("=" * 70)
        for src in (src_a, src_b):
            try:
                d = part_d_signal_diag(args.runs_root, src, args.target, args.model,
                                       fs_target, args.n_folds, args.pred_glob)
                print(f"--- {args.model}, {src}→{args.target}, n={len(d)}개 recording ---")
                if len(d):
                    print(d[["std", "dom_freq_hz", "cardiac_band_power_ratio"]]
                          .describe().to_string())
                d.to_csv(out_dir / f"d_signaldiag_{args.model}_{src}_to_{args.target}.csv",
                        index=False)
            except Exception as e_:
                print(f"[건너뜀] {src}→{args.target}: {e_}")
        print(f"\n-> {src_b}→{args.target}의 cardiac_band_power_ratio가 {src_a}→{args.target}보다")
        print(f"   뚜렷이 낮으면(=대역 밖으로 새는 에너지가 많으면) '예측 자체가 비주기적/붕괴'된 것.")
        print(f"   std가 비정상적으로 작으면(거의 flat) '평균으로 수렴하는 퇴화 예측'을 의심.")

    print(f"\n모든 중간 산출물 저장: {out_dir}/")
    print("\n[판단 가이드]")
    print("  실험 없이 결론 내릴 수 있는 최소 기준(권장):")
    print("   1) (A)에서 physdiff의 swing_rank가 1~2위로 확인 -> '진짜 physdiff-특이 현상'이라 부를 근거 확보")
    print("   2) (B)에서 붕괴 주도 컴포넌트를 특정 -> Discussion에 '어떤 실패모드'인지 구체적으로 서술 가능")
    print("   3) (C)(D) 중 하나라도 뚜렷한 원인이 잡히면 -> 그걸 근거로 '관찰됨+가설' 서술,")
    print("      셋 다 애매하면 -> '원인 미상, 향후 연구 과제'로 정직하게 남기고 추가 실험은 보류해도 무방")
    print("      (36건 대응이 이미 실험적으로 충분한 상황이므로, 이 항목은 '선택'대로 취급 가능)")


if __name__ == "__main__":
    main()

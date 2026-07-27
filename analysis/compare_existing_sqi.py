"""
compare_existing_sqi.py — 기존 PPG/rPPG SQI와 RPFI 직접 비교 (R2#2 대응)
=========================================================================
리뷰어 R2#2: "기존 복합지표(SQI 등)와의 비교가 부족하다."

핵심 프레이밍
-------------
기존 PPG Signal Quality Index(SQI)들은 대부분 **reference-free**다 — 신호
하나만 보고 "이게 깨끗한 PPG처럼 생겼는가"를 판단한다. RPFI는 **reference-based**다
— contact PPG(label)와 대조해서 "이게 실제로 맞는 파형인가"를 판단한다.
그래서 "RPFI가 SQI보다 우월하다"가 아니라 **"SQI가 못 보는 실패모드를 RPFI가
잡아낸다"**는 게 이 스크립트가 만들 증거의 요지다. 특히 "SQI는 높은데(신호가
깨끗해 보이는데) RPFI는 낮은(실제로는 틀린)" 사례가 있으면, 이미 확보한
RPFI-HR MAE 반례(HR은 맞는데 파형은 틀린 경우)와 같은 계열의 강력한 R2 대응
논거가 된다.

구현한 4개 고전 SQI (전부 예측 신호 **단독**으로만 계산, label 미사용 — 이게 핵심)
-------------------------------------------------------------------------------
  1. skewness SQI      — Sukor, Redmond & Lovell (2011), Physiol. Meas.
  2. kurtosis SQI       — Sukor et al. (2011); Selvaraj et al. (2011)
  3. relative-power SQI — Elgendi (2016), Bioengineering (1st+2nd harmonic 밴드 파워 비율)
  4. beat-template correlation SQI — Li & Clifford (2012) 계열, 인접 비트 형태 유사도
※ skewness/kurtosis는 문헌마다 "좋은 방향"에 대한 관례가 갈려서(논문마다 부호 해석이
  다름) 이 스크립트에서는 방향성을 강제하지 않고 서술적 지표로만 다룬다.
  "높을수록 청결하다"는 방향이 명확한 relpower/template-corr 두 개만 합성 품질
  프록시로 사용한다.

전제 (train_models.py 스키마에서 직접 확인한 필드 사용 — 정확함)
-------------------------------------------------------------
  runs/{dataset}_cv5/{model}_f{k}_info.json      : test_subject_of_range (seq_len은 선택적 —
                                                    train_bimamba_blend.py 산출물엔 없어서
                                                    fs*--eval-seconds로 직접 계산함, rpfi_eval.py와 동일 방식)
  runs/{dataset}_cv5/{model}_f{k}_test_pred.npy  : object array, recording별 1D, window-znorm
  결과 참여자 단위 RPFI: results_{dataset}_final/rpfi_participant_{dataset}_{agg}.csv
    (§4-c에서 확인된 경로. 컬럼명은 자동 탐지하되 못 찾으면 --agg/컬럼 관련 인자로 보정)

사용 예:
    python compare_existing_sqi.py --runs-root runs --results-root . \
        --datasets pure,ubfc,cohface --models all
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch, find_peaks
from scipy.stats import kurtosis, skew, spearmanr

CARDIAC_BAND = (0.75, 3.0)
FS = {"pure": 30, "ubfc": 30, "cohface": 20}
ALL_MODELS = ["pos", "rnn", "lstm", "bilstm", "physdiff", "transformer",
              "mamba", "bimamba", "bimamba_blend"]


# ══════════════════════════════════════════════════════════════
# SQI 구현체 4종 — 전부 예측 신호 x 하나만 입력받음 (label 사용 안 함)
# ══════════════════════════════════════════════════════════════

def skewness_sqi(x):
    if np.std(x) < 1e-8:
        return np.nan
    return float(skew(x))


def kurtosis_sqi(x):
    if np.std(x) < 1e-8:
        return np.nan
    return float(kurtosis(x))  # fisher=True (excess), 정규분포=0


def relpower_sqi(x, fs, band=CARDIAC_BAND, wide_band=(0.5, 8.0), tol_hz=0.2):
    """Elgendi(2016) 계열: 지배주파수(f0)와 그 2배음(2f0) 근방 파워 / 전체 파워.
    label 없이 예측 신호 스스로 '주기적인가'를 판단 -> RPFI의 SRE(SNR 추정)와
    개념적으로 유사하지만, RPFI의 SNR은 **label**에서 추정한다는 게 결정적 차이."""
    n = len(x)
    if n < fs * 2:
        return np.nan
    f, pxx = welch(x, fs=fs, nperseg=min(256, n))
    in_band = (f >= band[0]) & (f <= band[1])
    if not in_band.any() or pxx[in_band].sum() <= 0:
        return np.nan
    f0 = f[in_band][np.argmax(pxx[in_band])]
    nyq = fs / 2 - 1e-6
    wide = (f >= wide_band[0]) & (f <= min(wide_band[1], nyq))
    total = pxx[wide].sum()
    if total <= 0:
        return np.nan

    def band_power(center):
        m = (f >= center - tol_hz) & (f <= center + tol_hz)
        return pxx[m].sum()

    num = band_power(f0) + (band_power(2 * f0) if 2 * f0 <= nyq else 0.0)
    return float(np.clip(num / total, 0.0, 1.0))


def template_corr_sqi(x, fs, max_hr=180, resample_len=40, min_beats=3):
    """Li & Clifford(2012) 계열: 검출된 비트들을 고정 길이로 리샘플해
    비트간 상관계수의 평균을 품질 점수로 사용. label 미사용, 신호 자체의
    '자기일관성(self-consistency)'만 본다."""
    n = len(x)
    if n < fs * 2:
        return np.nan
    min_dist = max(1, int(fs * 60 / max_hr))
    peaks, _ = find_peaks(x, distance=min_dist)
    if len(peaks) < min_beats:
        return np.nan
    med_gap = int(np.median(np.diff(peaks)))
    half = max(2, med_gap // 2)
    beats = []
    for p in peaks:
        s, e = p - half, p + half
        if s < 0 or e > n:
            continue
        seg = x[s:e]
        if seg.size < 4:
            continue
        xs_old = np.linspace(0, 1, seg.size)
        xs_new = np.linspace(0, 1, resample_len)
        beats.append(np.interp(xs_new, xs_old, seg))
    if len(beats) < min_beats:
        return np.nan
    B = np.stack(beats)
    C = np.corrcoef(B)
    iu = np.triu_indices_from(C, k=1)
    vals = C[iu]
    vals = vals[~np.isnan(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


SQI_FUNCS_1ARG = {"skew_sqi": skewness_sqi, "kurt_sqi": kurtosis_sqi}
SQI_FUNCS_FS = {"relpower_sqi": relpower_sqi, "template_corr_sqi": template_corr_sqi}
DIRECTIONAL_SQI = ["relpower_sqi", "template_corr_sqi"]   # 높을수록 '청결' 방향이 명확한 것들


# ══════════════════════════════════════════════════════════════
# 예측 로드 (train_models.py 스키마 그대로 사용)
# ══════════════════════════════════════════════════════════════

def load_chunks_with_participants(runs_root, dataset, model, n_folds, seg_len,
                                  info_glob=None, pred_glob=None):
    """★ seg_len은 info.json의 'seq_len' 키에서 읽지 않는다 — train_bimamba_blend.py가
    저장하는 info.json에는 그 키가 없다(train_models.py/export_pos_baseline.py만 저장함).
    rpfi_eval.py도 같은 이유로 info.json을 안 보고 fs*eval_seconds로 직접 계산하므로
    이 함수도 그 방식을 그대로 따른다 (호출부에서 계산해 인자로 넘김)."""
    root = Path(runs_root) / f"{dataset}_cv{n_folds}"
    info_pat = info_glob or f"{model}_f{{k}}_info.json"
    pred_pat = pred_glob or f"{model}_f{{k}}_test_pred.npy"
    out = []
    for k in range(n_folds):
        info_f, pred_f = root / info_pat.format(k=k), root / pred_pat.format(k=k)
        if not info_f.exists() or not pred_f.exists():
            continue
        info = json.loads(info_f.read_text())
        subj_of_range = info["test_subject_of_range"]
        preds = np.load(pred_f, allow_pickle=True)
        if len(preds) != len(subj_of_range):
            warnings.warn(f"{dataset}/{model} f{k}: 예측 recording 수({len(preds)}) != "
                          f"test_subject_of_range 길이({len(subj_of_range)}) — 건너뜀")
            continue
        for rec_i, (arr, subj) in enumerate(zip(preds, subj_of_range)):
            arr = np.asarray(arr, dtype=np.float64)
            n_chunks = len(arr) // seg_len
            for c in range(n_chunks):
                out.append(dict(fold=k, participant=subj, recording=rec_i, chunk_idx=c,
                                chunk=arr[c * seg_len:(c + 1) * seg_len]))
    return out


def compute_sqi_table(runs_root, datasets, models, n_folds, eval_seconds, info_glob, pred_glob):
    rows = []
    for ds in datasets:
        fs = FS[ds]
        seg_len = int(round(fs * eval_seconds))
        for model in models:
            chunks = load_chunks_with_participants(runs_root, ds, model, n_folds, seg_len,
                                                    info_glob, pred_glob)
            if not chunks:
                warnings.warn(f"{ds}/{model}: 로드된 청크 없음 (파일 경로 확인 필요)")
                continue
            for rec in chunks:
                x = rec["chunk"]
                row = dict(dataset=ds, model=model, participant=rec["participant"],
                          fold=rec["fold"], recording=rec["recording"], chunk_idx=rec["chunk_idx"])
                for name, fn in SQI_FUNCS_1ARG.items():
                    row[name] = fn(x)
                for name, fn in SQI_FUNCS_FS.items():
                    row[name] = fn(x, fs)
                rows.append(row)
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 기존 RPFI participant CSV와 조인
# ══════════════════════════════════════════════════════════════

def find_col(df, keywords):
    """정확일치를 부분일치보다 먼저 본다 ('RPFI' vs 'RPFI_alt_agg' 혼동 방지)."""
    for kw in keywords:
        for c in df.columns:
            if c.lower() == kw.lower():
                return c
    for kw in keywords:
        for c in df.columns:
            if kw.lower() in c.lower():
                return c
    return None


def load_rpfi_participant(results_root, dataset, agg):
    d = Path(results_root) / f"results_{dataset}_final"
    cands = sorted(d.glob(f"rpfi_participant_{dataset}_{agg}*.csv"))
    if not cands:
        cands = sorted(d.glob("rpfi_participant_*.csv"))
    if not cands:
        existing = [p.name for p in d.glob("*")] if d.exists() else ["<폴더 없음>"]
        raise FileNotFoundError(f"{d}에 rpfi_participant_*.csv 없음. 실제 내용: {existing}")
    return pd.read_csv(cands[0]), cands[0]


def merge_sqi_rpfi(sqi_participant_df, results_root, datasets, agg):
    merged = []
    for ds in datasets:
        try:
            rpfi_df, path = load_rpfi_participant(results_root, ds, agg)
        except FileNotFoundError as e:
            print(f"[건너뜀] {e}")
            continue
        model_col = find_col(rpfi_df, ["model"])
        subj_col = find_col(rpfi_df, ["participant", "subject"])
        rpfi_col = find_col(rpfi_df, ["rpfi"])   # 정확일치 우선이라 'RPFI'(geometric)가 잡힘
        if not (model_col and subj_col and rpfi_col):
            print(f"[{ds}] 컬럼 자동탐지 실패. 실제 컬럼: {list(rpfi_df.columns)}")
            continue
        comp_cols = {c: find_col(rpfi_df, [c]) for c in ["bwmd", "sre", "wcr", "edd", "hr_mae"]}
        rn = {model_col: "model", subj_col: "participant", rpfi_col: "RPFI"}
        for k, v in comp_cols.items():
            if v:
                rn[v] = k.upper()
        r = rpfi_df.rename(columns=rn)
        r["dataset"] = ds
        r["model"] = r["model"].astype(str).str.lower()
        keep = ["dataset", "model", "participant", "RPFI"] + \
               [k.upper() for k, v in comp_cols.items() if v]
        merged.append(r[keep])

        sub = sqi_participant_df[sqi_participant_df["dataset"] == ds].copy()
        sub["model"] = sub["model"].astype(str).str.lower()
        # participant ID 표기가 다를 수 있어 느슨한 정규화 시도 (접두사/대소문자 무시)
        def norm(s):
            return str(s).lower().replace(f"{ds}_", "").strip()
        sub["_pkey"] = sub["participant"].map(norm)
        r["_pkey"] = r["participant"].astype(str).map(norm)
        if not (set(sub["_pkey"]) & set(r["_pkey"])):
            print(f"[{ds}] 경고: participant ID가 두 소스에서 하나도 안 겹칩니다.")
            print(f"   SQI측 예시: {sub['participant'].unique()[:5]}")
            print(f"   RPFI측 예시: {r['participant'].unique()[:5]}")

    if not merged:
        raise RuntimeError("RPFI CSV를 하나도 못 불러왔습니다. --results-root/--agg를 확인하세요.")
    rpfi_all = pd.concat(merged, ignore_index=True)

    sqi_p = (sqi_participant_df.groupby(["dataset", "model", "participant"])
            [list(SQI_FUNCS_1ARG) + list(SQI_FUNCS_FS)].mean().reset_index())
    sqi_p["model"] = sqi_p["model"].astype(str).str.lower()

    def norm_all(df, ds_col="dataset", p_col="participant"):
        return df.apply(lambda row: str(row[p_col]).lower().replace(f"{row[ds_col]}_", "").strip(),
                        axis=1)
    sqi_p["_pkey"] = norm_all(sqi_p)
    rpfi_all["_pkey"] = norm_all(rpfi_all)

    out = pd.merge(sqi_p, rpfi_all, on=["dataset", "model", "_pkey"],
                   how="inner", suffixes=("", "_rpfi"))
    return out.drop(columns=["_pkey"])


# ══════════════════════════════════════════════════════════════
# 분석: 상관 + SQI 사각지대
# ══════════════════════════════════════════════════════════════

def correlation_report(merged, pooled=True):
    sqi_cols = list(SQI_FUNCS_1ARG) + list(SQI_FUNCS_FS)
    target_cols = [c for c in ["RPFI", "BWMD", "SRE", "WCR", "EDD", "HR_MAE"] if c in merged.columns]
    rows = []
    groups = [("ALL", merged)] if pooled else [(ds, g) for ds, g in merged.groupby("dataset")]
    for grp_name, g in groups:
        for sqi in sqi_cols:
            for tgt in target_cols:
                sub = g[[sqi, tgt]].dropna()
                if len(sub) < 5:
                    continue
                rho, p = spearmanr(sub[sqi], sub[tgt])
                rows.append(dict(group=grp_name, sqi=sqi, target=tgt,
                                 spearman_rho=round(rho, 3), p=p, n=len(sub)))
    return pd.DataFrame(rows)


def sqi_blindspot_cases(merged, top_q=0.75, bottom_q=0.25, min_n=20):
    """SQI(청결해 보임, 상위 25%)는 높은데 RPFI(실제로는 틀림, 하위 25%)는 낮은 사례.
    RPFI-HR MAE 반례와 같은 계열의 R2#2 대응 논거."""
    d = merged.copy()
    avail = [c for c in DIRECTIONAL_SQI if c in d.columns]
    if not avail or "RPFI" not in d.columns:
        return pd.DataFrame(), None
    for c in avail:
        d[f"{c}_pct"] = d[c].rank(pct=True)
    d["composite_sqi_pct"] = d[[f"{c}_pct" for c in avail]].mean(axis=1)
    d["rpfi_pct"] = d["RPFI"].rank(pct=True)
    blind = d[(d["composite_sqi_pct"] >= top_q) & (d["rpfi_pct"] <= bottom_q)]
    frac = len(blind) / len(d) if len(d) else float("nan")
    cols = ["dataset", "model", "participant", "RPFI", "composite_sqi_pct"] + avail
    return blind[cols].sort_values("RPFI"), frac


# ══════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--results-root", default=".")
    ap.add_argument("--datasets", default="pure,ubfc,cohface")
    ap.add_argument("--models", default="all",
                    help="콤마 구분 모델 리스트 또는 'all' "
                        f"(all = {ALL_MODELS})")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--eval-seconds", type=float, default=10.0,
                    help="rpfi_eval.py --eval-seconds와 동일 (기본 10초)")
    ap.add_argument("--agg", default="geometric")
    ap.add_argument("--info-glob", default=None)
    ap.add_argument("--pred-glob", default=None)
    ap.add_argument("--out", default="sqi_comparison_report")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]
    models = ALL_MODELS if args.models == "all" else [m.strip() for m in args.models.split(",")]

    out_dir = Path(args.out); out_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 70)
    print("1단계: 예측 신호에서 4종 SQI 계산 (label 미사용)")
    print("=" * 70)
    sqi_chunks = compute_sqi_table(args.runs_root, datasets, models, args.n_folds,
                                   args.eval_seconds, args.info_glob, args.pred_glob)
    if sqi_chunks.empty:
        raise SystemExit("SQI를 하나도 계산 못 했습니다. --runs-root 경로/파일명 컨벤션을 확인하세요.")
    print(f"총 {len(sqi_chunks):,}개 청크에서 SQI 계산 완료 "
         f"({sqi_chunks['dataset'].nunique()}개 데이터셋 x {sqi_chunks['model'].nunique()}개 모델)")
    sqi_chunks.to_csv(out_dir / "sqi_chunk_level.csv", index=False)

    print("\n" + "=" * 70)
    print("2단계: 기존 RPFI participant CSV와 조인")
    print("=" * 70)
    merged = merge_sqi_rpfi(sqi_chunks, args.results_root, datasets, args.agg)
    print(f"조인 결과: {len(merged):,}개 participant-model 행")
    merged.to_csv(out_dir / "sqi_rpfi_merged.csv", index=False)
    if merged.empty:
        raise SystemExit("조인 결과가 0행입니다. participant ID 표기 불일치일 가능성이 큽니다 "
                         "(위 경고 메시지의 예시 ID를 비교해 --results-root 등을 확인하세요).")

    print("\n" + "=" * 70)
    print("3단계: SQI vs RPFI(+컴포넌트) 상관 (Spearman, 데이터셋 pooled)")
    print("=" * 70)
    corr = correlation_report(merged, pooled=True)
    print(corr.to_string(index=False))
    corr.to_csv(out_dir / "correlation_pooled.csv", index=False)

    print("\n--- 데이터셋별 ---")
    corr_by_ds = correlation_report(merged, pooled=False)
    print(corr_by_ds.to_string(index=False))
    corr_by_ds.to_csv(out_dir / "correlation_by_dataset.csv", index=False)

    print("\n" + "=" * 70)
    print("4단계: SQI 사각지대 — '깨끗해 보이는데 실제론 틀린' 사례")
    print("=" * 70)
    blind, frac = sqi_blindspot_cases(merged)
    if frac is not None:
        print(f"전체 {len(merged)}건 중 {len(blind)}건 ({frac:.1%})이 "
             f"composite_sqi 상위25% & RPFI 하위25%에 동시 해당")
        if len(blind):
            print(blind.head(15).to_string(index=False))
        blind.to_csv(out_dir / "sqi_blindspot_cases.csv", index=False)
    else:
        print("directional SQI 또는 RPFI 컬럼이 부족해 계산 불가")

    print(f"\n모든 산출물 저장: {out_dir}/")
    print("\n[response letter/Discussion에 쓸 수 있는 포인트]")
    print("  - relpower/template-corr SQI와 RPFI의 상관이 약~중간(예: |rho|<0.5)이면:")
    print("    '기존 reference-free SQI와 RPFI는 상당 부분 독립 정보 -> RPFI가 SQI의")
    print("     대체재가 아니라 보완재'라는 R2#2 대응 논거로 직접 사용 가능")
    print("  - blindspot 비율이 유의미하게 존재하면(예: >5%): RPFI-HR MAE 반례와")
    print("    같은 계열로 'SQI가 못 잡는 실패모드를 RPFI가 잡아낸다'는 정량적 증거")
    print("  - skew/kurtosis는 방향성 강제 안 했으므로 상관표만 참고용으로 보고,")
    print("    본문에는 상관계수 자체보다 '무엇을 측정하는 지표인지'가 다르다는")
    print("    개념적 차이(reference-free vs reference-based)를 주 논거로 쓸 것")


if __name__ == "__main__":
    main()

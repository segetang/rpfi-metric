"""
subject_split_v2.py — 공저자 제공 매핑 CSV 기반 subject-wise split
====================================================================
v1(subject_split.py)은 main_eval_final.py의 subject_counts를 사용했으나,
그 목록은 **recording 단위**였고 COHFACE는 배열 길이와 불일치했다.
v2는 공저자가 제공한 mapping_*.csv를 사용하여 **실제 participant 단위**로 분할한다.

매핑 검증 결과 (실제 npy와 대조 완료)
-------------------------------------
  PURE   : 59 recording / 10 subject / 113,400 frame  ✓ 배열과 일치
  UBFC   : 40 recording / 40 subject /  69,300 frame  ✓ (1인 1영상)
  COHFACE: 143 recording / 40 subject / 173,100 frame ✓
           └ recovered 132개(159,300 frame) + unknown 11개(13,800 frame, 8.0%)

주의: COHFACE의 unknown 11개는 subject를 특정할 수 없으므로 **제외**한다.
      (test subject의 데이터가 train에 섞일 위험을 배제할 수 없음)

세그먼트 길이
-------------
아카이브는 세 데이터셋 모두 **300 프레임** 단위로 분할되어 있다.
  PURE/UBFC : 300 frame @ 30 Hz = 10 초
  COHFACE   : 300 frame @ 20 Hz = 15 초   ← 논문의 "10초(200샘플)" 기술과 불일치

--chunk-mode 로 선택:
  archive : 300 프레임 고정 (아카이브 원형 유지, COHFACE는 15초)
  seconds : 초 단위 통일 (--chunk-sec 10 이면 COHFACE는 200 프레임, 나머지 300)

사용법
------
    python subject_split_v2.py --dataset all --mapping-dir . --out-dir splits
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

FS = {"pure": 30, "ubfc": 30, "cohface": 20}
MAPPING_FILE = {"pure": "mapping_PURE.csv",
                "ubfc": "mapping_UBFC.csv",
                "cohface": "mapping_COHFACE.csv"}
ARCHIVE_SEG = 300


def load_mapping(dataset, mapping_dir):
    """매핑 CSV -> recording 리스트.
    각 항목: dict(subject, start, end, n_frames, usable)
    """
    path = Path(mapping_dir) / MAPPING_FILE[dataset]
    rows = list(csv.DictReader(open(path)))
    recs = []
    for r in rows:
        if dataset == "cohface":
            subj = (r.get("subject") or "").strip()
            usable = (r.get("status", "").strip() == "recovered") and subj != ""
            sid = f"cohface_{subj}" if subj else None
        else:
            subj = (r.get("subject_id") or "").strip()
            usable = subj != ""
            sid = f"{dataset}_{subj}"
        recs.append(dict(subject=sid,
                         start=int(r["frame_start"]), end=int(r["frame_end"]),
                         n_frames=int(r["n_frames"]), usable=usable))
    return recs


def make_split(recs, fs, test_ratio=0.2, seed=42, chunk_mode="archive", chunk_sec=10,
               dataset=None):
    """participant 단위 train/test 분할."""
    usable = [r for r in recs if r["usable"]]
    dropped = [r for r in recs if not r["usable"]]

    subjects = sorted({r["subject"] for r in usable})
    rng = np.random.default_rng(seed)
    perm = np.array(subjects, dtype=object)
    rng.shuffle(perm)
    n_test = max(1, int(round(len(subjects) * test_ratio)))
    test_subj = set(perm[:n_test].tolist())
    train_subj = set(perm[n_test:].tolist())

    seg = ARCHIVE_SEG if chunk_mode == "archive" else int(fs * chunk_sec)

    def pack(sel):
        out = []
        for r in usable:
            if r["subject"] in sel:
                # recording 내부에서 seg 배수만큼만 사용 (경계 넘는 윈도우 방지)
                n = (r["n_frames"] // seg) * seg
                if n > 0:
                    out.append(dict(subject=r["subject"],
                                    start=r["start"], end=r["start"] + n))
        return out

    tr, te = pack(train_subj), pack(test_subj)
    return {
        "dataset": dataset,
        "n_frames_total": sum(r["n_frames"] for r in recs),
        "seed": seed, "fs": fs, "chunk_mode": chunk_mode,
        "seg_len": seg,
        "seg_seconds": round(seg / fs, 3),
        "test_ratio": test_ratio,
        "n_subjects_total": len(subjects),
        "n_subjects_train": len(train_subj),
        "n_subjects_test": len(test_subj),
        "train_subjects": sorted(train_subj),
        "test_subjects": sorted(test_subj),
        "train_ranges": [(d["start"], d["end"]) for d in tr],
        "test_ranges": [(d["start"], d["end"]) for d in te],
        "train_subject_of_range": [d["subject"] for d in tr],
        "test_subject_of_range": [d["subject"] for d in te],
        "n_dropped_recordings": len(dropped),
        "dropped_frames": sum(r["n_frames"] for r in dropped),
    }


def make_cv_folds(recs, fs, K=5, seed=42, chunk_mode="seconds", chunk_sec=10,
                  dataset=None):
    """participant 단위 K-fold cross-validation.

    K = subject 수 이면 leave-one-subject-out(LOSO).
    fold 배정은 셔플 후 round-robin (fold 크기 균형).
    리뷰어 R1-12의 "repeated subject-level cross-validation" 요구에 대응.
    """
    usable = [r for r in recs if r["usable"]]
    dropped = [r for r in recs if not r["usable"]]
    subjects = sorted({r["subject"] for r in usable})

    rng = np.random.default_rng(seed)
    perm = np.array(subjects, dtype=object)
    rng.shuffle(perm)
    K = min(K, len(subjects))
    fold_subj = [sorted(perm[i::K].tolist()) for i in range(K)]

    seg = ARCHIVE_SEG if chunk_mode == "archive" else int(fs * chunk_sec)

    def pack(sel):
        out = []
        for r in usable:
            if r["subject"] in sel:
                n = (r["n_frames"] // seg) * seg
                if n > 0:
                    out.append((r["subject"], r["start"], r["start"] + n))
        return out

    folds = []
    for k in range(K):
        te_s = set(fold_subj[k])
        tr_s = set(subjects) - te_s
        tr, te = pack(tr_s), pack(te_s)
        folds.append({
            "fold": k,
            "train_subjects": sorted(tr_s), "test_subjects": sorted(te_s),
            "train_ranges": [(s, e) for _, s, e in tr],
            "test_ranges": [(s, e) for _, s, e in te],
            "train_subject_of_range": [sub for sub, _, _ in tr],
            "test_subject_of_range": [sub for sub, _, _ in te],
        })

    return {
        "dataset": dataset,
        "n_frames_total": sum(r["n_frames"] for r in recs),
        "mode": "cv", "K": K, "seed": seed, "fs": fs,
        "chunk_mode": chunk_mode, "seg_len": seg,
        "seg_seconds": round(seg / fs, 3),
        "n_subjects_total": len(subjects),
        "fold_sizes": [len(f) for f in fold_subj],
        "n_dropped_recordings": len(dropped),
        "dropped_frames": sum(r["n_frames"] for r in dropped),
        "folds": folds,
    }


def load_cv(path):
    d = json.load(open(path))
    for f in d["folds"]:
        f["train_ranges"] = [tuple(r) for r in f["train_ranges"]]
        f["test_ranges"] = [tuple(r) for r in f["test_ranges"]]
    return d


def save(split, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(path, "w"), indent=2)


def load_split(path):
    d = json.load(open(path))
    d["train_ranges"] = [tuple(r) for r in d["train_ranges"]]
    d["test_ranges"] = [tuple(r) for r in d["test_ranges"]]
    return d


def verify_split_matches(split, arr, dataset_arg=None, what="배열"):
    """split JSON과 실제 배열이 같은 데이터셋인지 검증한다.

    ★ 이 검증이 없으면 다른 데이터셋의 split을 써도 에러 없이 조용히
      엉뚱한 구간을 잘라내어 subject-disjoint 보장이 깨진다.
    """
    ds = split.get("dataset")
    tot = split.get("n_frames_total")
    if dataset_arg and ds and ds != dataset_arg:
        raise ValueError(
            f"split의 dataset('{ds}')과 --dataset('{dataset_arg}')이 다릅니다. "
            f"잘못된 split 파일을 지정했을 가능성이 큽니다.")
    if tot is not None and len(arr) != tot:
        raise ValueError(
            f"{what} 길이({len(arr):,})가 split의 n_frames_total({tot:,})과 다릅니다. "
            f"데이터셋과 split이 짝이 맞지 않습니다. "
            f"(split dataset='{ds}')")
    return True


def extract_by_ranges(arr, ranges):
    return [np.asarray(arr[s:e]) for s, e in ranges]


def make_windows_per_subject(x_list, y_list, seq_len, stride=None):
    """recording 경계를 넘지 않는 윈도우 생성.
    stride=None이면 non-overlapping(=seq_len)."""
    stride = stride or seq_len
    xs, ys = [], []
    for xs_, ys_ in zip(x_list, y_list):
        n = min(len(xs_), len(ys_))
        for i in range(0, n - seq_len + 1, stride):
            xs.append(xs_[i:i + seq_len]); ys.append(ys_[i:i + seq_len])
    if not xs:
        raise ValueError("윈도우가 생성되지 않았습니다.")
    return np.stack(xs), np.stack(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pure", "ubfc", "cohface", "all"], default="all")
    ap.add_argument("--mapping-dir", default=".")
    ap.add_argument("--out-dir", default="splits")
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-mode", choices=["archive", "seconds"], default="seconds")
    ap.add_argument("--chunk-sec", type=int, default=10)
    ap.add_argument("--cv", type=int, default=0,
                    help="K-fold subject CV 생성 (0=단일 split). "
                         "K가 subject 수 이상이면 LOSO")
    args = ap.parse_args()

    targets = ["pure", "ubfc", "cohface"] if args.dataset == "all" else [args.dataset]

    if args.cv:
        for ds in targets:
            recs = load_mapping(ds, args.mapping_dir)
            cv = make_cv_folds(recs, FS[ds], args.cv, args.seed,
                               args.chunk_mode, args.chunk_sec, dataset=ds)
            out = Path(args.out_dir) / f"{ds}_cv{cv['K']}.json"
            save(cv, out)
            # 무결성: 모든 fold에서 train/test subject 교집합 없음
            for f in cv["folds"]:
                assert not (set(f["train_subjects"]) & set(f["test_subjects"]))
            loso = " (LOSO)" if cv["K"] == cv["n_subjects_total"] else ""
            print(f"[{ds}] {cv['K']}-fold subject CV{loso}  "
                  f"seg_len={cv['seg_len']} ({cv['seg_seconds']}s)")
            print(f"   subject {cv['n_subjects_total']}명, fold 크기 {cv['fold_sizes']}")
            if cv["n_dropped_recordings"]:
                print(f"   ⚠️ 제외 recording {cv['n_dropped_recordings']}개 "
                      f"({cv['dropped_frames']:,} frame)")
            print(f"   -> {out}")
        return

    for ds in targets:
        recs = load_mapping(ds, args.mapping_dir)
        sp = make_split(recs, FS[ds], args.test_ratio, args.seed,
                        args.chunk_mode, args.chunk_sec, dataset=ds)
        out = Path(args.out_dir) / f"{ds}_split.json"
        save(sp, out)

        tr_n = sum(e - s for s, e in sp["train_ranges"])
        te_n = sum(e - s for s, e in sp["test_ranges"])
        # 무결성 검사: subject가 양쪽에 걸치지 않는지
        assert not (set(sp["train_subjects"]) & set(sp["test_subjects"])), "subject 중복!"
        allr = sorted(sp["train_ranges"] + sp["test_ranges"])
        for (s1, e1), (s2, e2) in zip(allr, allr[1:]):
            assert e1 <= s2, f"구간 겹침: ({s1},{e1}) vs ({s2},{e2})"

        print(f"[{ds}] seg_len={sp['seg_len']} ({sp['seg_seconds']}s @ {FS[ds]}Hz)")
        print(f"   subject {sp['n_subjects_train']} train / {sp['n_subjects_test']} test "
              f"(총 {sp['n_subjects_total']})")
        print(f"   frame   {tr_n:,} train / {te_n:,} test  (test 비율 {te_n/(tr_n+te_n):.3f})")
        if sp["n_dropped_recordings"]:
            print(f"   ⚠️ 제외된 recording {sp['n_dropped_recordings']}개 "
                  f"({sp['dropped_frames']:,} frame) — subject 미상")
        print(f"   -> {out}")


if __name__ == "__main__":
    main()

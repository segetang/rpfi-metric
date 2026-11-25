"""
subject_split_v2.py — subject-wise splits based on the co-author-provided mapping CSVs
====================================================================
v1 (subject_split.py) used subject_counts from main_eval_final.py, but that list was
**per recording** and did not match the array length for COHFACE.
v2 splits by **actual participant** using the mapping_*.csv files provided by a co-author.

Mapping check (verified against the actual npy arrays)
-------------------------------------
  PURE   : 59 recordings / 10 subjects / 113,400 frames  [ok] matches the array
  UBFC   : 40 recordings / 40 subjects /  69,300 frames  [ok] (one video per person)
  COHFACE: 143 recordings / 40 subjects / 173,100 frames [ok]
           - 132 recovered (159,300 frames) + 11 unknown (13,800 frames, 8.0%)

Note: the 11 COHFACE recordings with unknown subject cannot be attributed and are **excluded**
      (the risk of a test subject's data leaking into training cannot be ruled out).

Segment length
-------------
The archive is segmented into **300-frame** units for all three datasets.
  PURE/UBFC : 300 frames @ 30 Hz = 10 s
  COHFACE   : 300 frames @ 20 Hz = 15 s

Selected with --chunk-mode:
  archive : fixed 300 frames (archive layout kept; 15 s for COHFACE)
  seconds : unified in seconds (with --chunk-sec 10, 200 frames for COHFACE and 300 for the others)

Usage
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
    """Mapping CSV -> list of recordings.
    Each item: dict(subject, start, end, n_frames, usable)
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
    """Participant-level train/test split."""
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
                # within a recording, use only a multiple of seg (no window crosses a boundary)
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
    """Participant-level K-fold cross-validation.

    K = number of subjects gives leave-one-subject-out (LOSO).
    Folds are assigned round-robin after shuffling (balanced fold sizes).
    Addresses the request for "repeated subject-level cross-validation".
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


def verify_split_matches(split, arr, dataset_arg=None, what="array"):
    """Check that the split JSON and the actual array belong to the same dataset.

    Without this check, using another dataset's split would silently cut
      the wrong ranges and break the subject-disjoint guarantee.
    """
    ds = split.get("dataset")
    tot = split.get("n_frames_total")
    if dataset_arg and ds and ds != dataset_arg:
        raise ValueError(
            f"split dataset ('{ds}') differs from --dataset ('{dataset_arg}'). "
            f"The wrong split file was probably given.")
    if tot is not None and len(arr) != tot:
        raise ValueError(
            f"{what} length ({len(arr):,}) differs from the split's n_frames_total ({tot:,}). "
            f"Dataset and split do not match. "
            f"(split dataset='{ds}')")
    return True


def extract_by_ranges(arr, ranges):
    return [np.asarray(arr[s:e]) for s, e in ranges]


def make_windows_per_subject(x_list, y_list, seq_len, stride=None):
    """Generate windows that do not cross recording boundaries.
    stride=None means non-overlapping (= seq_len)."""
    stride = stride or seq_len
    xs, ys = [], []
    for xs_, ys_ in zip(x_list, y_list):
        n = min(len(xs_), len(ys_))
        for i in range(0, n - seq_len + 1, stride):
            xs.append(xs_[i:i + seq_len]); ys.append(ys_[i:i + seq_len])
    if not xs:
        raise ValueError("no windows were generated.")
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
                    help="generate K-fold subject CV (0 = single split). "
                         "K >= number of subjects gives LOSO")
    args = ap.parse_args()

    targets = ["pure", "ubfc", "cohface"] if args.dataset == "all" else [args.dataset]

    if args.cv:
        for ds in targets:
            recs = load_mapping(ds, args.mapping_dir)
            cv = make_cv_folds(recs, FS[ds], args.cv, args.seed,
                               args.chunk_mode, args.chunk_sec, dataset=ds)
            out = Path(args.out_dir) / f"{ds}_cv{cv['K']}.json"
            save(cv, out)
            # integrity: no train/test subject overlap in any fold
            for f in cv["folds"]:
                assert not (set(f["train_subjects"]) & set(f["test_subjects"]))
            loso = " (LOSO)" if cv["K"] == cv["n_subjects_total"] else ""
            print(f"[{ds}] {cv['K']}-fold subject CV{loso}  "
                  f"seg_len={cv['seg_len']} ({cv['seg_seconds']}s)")
            print(f"   {cv['n_subjects_total']} subjects, fold sizes {cv['fold_sizes']}")
            if cv["n_dropped_recordings"]:
                print(f"   [!] excluded recordings: {cv['n_dropped_recordings']} "
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
        # integrity check: no subject on both sides
        assert not (set(sp["train_subjects"]) & set(sp["test_subjects"])), "subject overlap!"
        allr = sorted(sp["train_ranges"] + sp["test_ranges"])
        for (s1, e1), (s2, e2) in zip(allr, allr[1:]):
            assert e1 <= s2, f"overlapping ranges: ({s1},{e1}) vs ({s2},{e2})"

        print(f"[{ds}] seg_len={sp['seg_len']} ({sp['seg_seconds']}s @ {FS[ds]}Hz)")
        print(f"   subject {sp['n_subjects_train']} train / {sp['n_subjects_test']} test "
              f"(total {sp['n_subjects_total']})")
        print(f"   frame   {tr_n:,} train / {te_n:,} test  (test fraction {te_n/(tr_n+te_n):.3f})")
        if sp["n_dropped_recordings"]:
            print(f"   [!] excluded recordings: {sp['n_dropped_recordings']} "
                  f"({sp['dropped_frames']:,} frame) - subject unknown")
        print(f"   -> {out}")


if __name__ == "__main__":
    main()

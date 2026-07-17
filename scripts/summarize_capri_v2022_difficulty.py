#!/usr/bin/env python3
"""Summarize TraDock CAPRI results on CAPRI Score v2022 Difficult / Easy splits.

Uses Shirali et al. target lists (T###-N -> S-T###.N) and compares against
precomputed scorer columns in their scores&labels CSVs when provided.

Important: TraDock and Shirali CSVs generally use different decoy pools for the
same target name (different model counts / identification codes). Target-level
comparison is useful, but is not identical to scoring the exact same decoys.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

GOOD = {"acceptable", "medium", "high"}
TOPKS = (1, 5, 10, 25, 100, 200)

# Higher score better unless listed here as lower-better.
LOWER_BETTER = {
    "PIsToN",
    "HADDOCK",
    "AP_PISA",
    "FIREDOCK",
    "PYDOCK_TOT",
    "ZRANK2",
    "ROSETTADOCK",
}

METHODS = [
    "PIsToN",
    "dMaSIF",
    "DeepRank-GNN",
    "GNN-DOVE",
    "HADDOCK",
    "AP_PISA",
    "CP_PIE",
    "FIREDOCK",
    "PYDOCK_TOT",
    "ZRANK2",
    "ROSETTADOCK",
    "SIPPER",
]


def target_id_to_st(target_id: str) -> str:
    m = re.fullmatch(r"T(\d+)-(\d+)", str(target_id).strip())
    if not m:
        raise ValueError(f"unexpected target_id: {target_id}")
    return f"S-T{m.group(1)}.{m.group(2)}"


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def success_at(labels: list[bool], k: int) -> int:
    return int(any(labels[: min(k, len(labels))]))


def auc_binary(y: list[int], s: list[float]) -> float:
    if len(set(y)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def eval_tradock(detail_rows: list[dict], targets: set[str]) -> dict:
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in detail_rows:
        t = r.get("target", "")
        if t in targets:
            by_t[t].append(r)

    hits = {k: 0 for k in TOPKS}
    aucs = []
    per = []
    for t in sorted(by_t):
        rows = sorted(by_t[t], key=lambda r: to_float(r.get("score"), -1e9), reverse=True)
        labels = [str(r.get("classification", "")).lower() in GOOD for r in rows]
        y = [1 if x else 0 for x in labels]
        s = [to_float(r.get("score"), 0.0) for r in rows]
        a = auc_binary(y, s)
        if a == a:
            aucs.append(a)
        for k in TOPKS:
            hits[k] += success_at(labels, k)
        per.append(
            {
                "target": t,
                "n_models": len(rows),
                "n_pos": sum(y),
                "auc": a,
                **{f"s@{k}": success_at(labels, k) for k in TOPKS},
            }
        )

    n = len(by_t)
    return {
        "method": "TraDock",
        "n": n,
        "mean_AUC": sum(aucs) / len(aucs) if aucs else float("nan"),
        "per_target": per,
        **{f"Success@{k}": (100.0 * hits[k] / n) if n else 0.0 for k in TOPKS},
    }


def eval_shirali(rows: list[dict], keep_ids: set[str], method: str) -> dict:
    by_t: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        tid = r.get("target_id", "")
        if tid not in keep_ids:
            continue
        by_t[tid].append(r)

    higher = method not in LOWER_BETTER
    hits = {k: 0 for k in TOPKS}
    aucs = []
    for tid in by_t:
        trows = [r for r in by_t[tid] if r.get(method) not in (None, "")]
        trows.sort(key=lambda r: to_float(r.get(method), 0.0), reverse=higher)
        labels = [int(float(r.get("label", 0) or 0)) == 1 for r in trows]
        y = [1 if x else 0 for x in labels]
        s = [to_float(r.get(method), 0.0) for r in trows]
        score_for_auc = s if higher else [-x for x in s]
        a = auc_binary(y, score_for_auc)
        if a == a:
            aucs.append(a)
        for k in TOPKS:
            hits[k] += success_at(labels, k)
    n = len(by_t)
    return {
        "method": method,
        "n": n,
        "mean_AUC": sum(aucs) / len(aucs) if aucs else float("nan"),
        **{f"Success@{k}": (100.0 * hits[k] / n) if n else 0.0 for k in TOPKS},
    }


def write_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "n", "mean_AUC"] + [f"Success@{k}" for k in TOPKS]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in fields}
            if isinstance(out["mean_AUC"], float):
                out["mean_AUC"] = f"{out['mean_AUC']:.4f}"
            for k in TOPKS:
                key = f"Success@{k}"
                if isinstance(out[key], float):
                    out[key] = f"{out[key]:.2f}"
            w.writerow(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tradock_detail", required=True, help="TraDock CAPRI detail CSV")
    p.add_argument("--difficult_csv", required=True, help="Shirali Difficult scores&labels CSV")
    p.add_argument("--easy_csv", required=True, help="Shirali Easy scores&labels CSV")
    p.add_argument("--out_dir", required=True, help="Output directory")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = load_csv(Path(args.tradock_detail))
    tradock_targets = {r["target"] for r in detail}

    for split, csv_path in (("difficult", args.difficult_csv), ("easy", args.easy_csv)):
        shirali = load_csv(Path(csv_path))
        tid_to_st = {}
        for r in shirali:
            tid = r["target_id"]
            tid_to_st[tid] = target_id_to_st(tid)
        all_st = sorted(set(tid_to_st.values()))
        overlap_st = sorted(set(all_st) & tradock_targets)
        missing_st = sorted(set(all_st) - tradock_targets)
        keep_ids = {tid for tid, st in tid_to_st.items() if st in overlap_st}

        list_path = out_dir / f"capri_v2022_{split}_targets.txt"
        list_path.write_text("\n".join(all_st) + "\n", encoding="utf-8")

        td = eval_tradock(detail, set(overlap_st))
        rows = [td]
        for m in METHODS:
            if m in shirali[0]:
                rows.append(eval_shirali(shirali, keep_ids, m))
        rows.sort(
            key=lambda r: (r.get("Success@1", 0), r.get("Success@10", 0), r.get("mean_AUC", 0) or 0),
            reverse=True,
        )
        write_table(out_dir / f"capri_v2022_{split}_ranking.csv", rows)

        meta = out_dir / f"capri_v2022_{split}_meta.txt"
        meta.write_text(
            "\n".join(
                [
                    f"split={split}",
                    f"shirali_targets={len(all_st)}",
                    f"tradock_overlap={len(overlap_st)}",
                    f"missing={len(missing_st)}",
                    f"missing_targets={', '.join(missing_st)}",
                    f"tradock_Success@1={td['Success@1']:.2f}",
                    f"tradock_Success@10={td['Success@10']:.2f}",
                    f"tradock_mean_AUC={td['mean_AUC']:.4f}",
                    "NOTE: TraDock and Shirali methods may use different decoy pools per target.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"[{split}] overlap={len(overlap_st)} missing={len(missing_st)}")
        print(f"  TraDock S@1={td['Success@1']:.1f}% S@10={td['Success@10']:.1f}%")
        print(f"  wrote {out_dir / f'capri_v2022_{split}_ranking.csv'}")


if __name__ == "__main__":
    main()

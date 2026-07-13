#!/usr/bin/env python3
"""Fill failed DeepRank-GNN-esm CAPRI targets.

DeepRank-GNN-esm can fail a whole multi-model target when a few decoys have no
valid graph/node attributes. This helper reruns the target after removing those
bad model indices, then writes a complete prediction CSV where the removed
models get a very low score.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", required=True, help="DeepRank CAPRI work dir, e.g. capri113_full")
    parser.add_argument("--target", action="append", default=None, help="Failed target to fill; can be repeated")
    parser.add_argument("--log", default=None, help="Full run log used to parse bad model indices")
    parser.add_argument("--ncores", type=int, default=8)
    parser.add_argument("--bad_score", type=float, default=-1.0e9)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_failed_targets(work_dir: Path) -> list[str]:
    failed_path = work_dir / "failed_targets.txt"
    if not failed_path.exists():
        return []
    return [line.strip() for line in failed_path.read_text().splitlines() if line.strip()]


def parse_bad_indices(log_path: Path | None, target: str) -> set[int]:
    if log_path is None or not log_path.exists():
        return set()
    text = log_path.read_text(errors="ignore")
    escaped = re.escape(target)
    patterns = [
        rf"{escaped}_model_(\d+)\.pkl",
        rf"deleting {escaped}_model_(\d+)",
    ]
    bad: set[int] = set()
    for pattern in patterns:
        bad.update(int(m.group(1)) for m in re.finditer(pattern, text))
    return bad


def split_models(pdb_path: Path) -> list[list[str]]:
    models: list[list[str]] = []
    current: list[str] | None = None
    with pdb_path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                current = [line]
            elif line.startswith("ENDMDL"):
                if current is not None:
                    current.append(line)
                    models.append(current)
                    current = None
            elif current is not None:
                current.append(line)

    if models:
        return models

    lines = [
        line
        for line in pdb_path.read_text(errors="ignore").splitlines(keepends=True)
        if line.startswith(("ATOM  ", "HETATM", "TER"))
    ]
    if lines:
        return [["MODEL        1\n", *lines, "ENDMDL\n"]]
    raise ValueError(f"no models found in {pdb_path}")


def write_filtered_pdb(models: list[list[str]], keep_indices: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for local_serial, original_idx in enumerate(keep_indices, start=1):
            lines = models[original_idx]
            if lines and lines[0].startswith("MODEL"):
                handle.write(f"MODEL     {local_serial:4d}\n")
                handle.writelines(lines[1:])
            else:
                handle.write(f"MODEL     {local_serial:4d}\n")
                handle.writelines(lines)
                handle.write("ENDMDL\n")
        handle.write("END\n")


def parse_prediction_csv(path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = row.get("pdb_id", "")
            match = re.search(r"_model_(\d+)$", pdb_id)
            if not match:
                continue
            out[int(match.group(1))] = float(row["predicted_fnat"])
    return out


def run_filtered(target: str, filtered_pdb: Path, work_dir: Path, ncores: int, force: bool) -> Path:
    run_root = work_dir / "fallback_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    workspace = run_root / f"{filtered_pdb.stem}-gnn_esm_pred_A_B"
    pred_csv = workspace / "GNN_esm_prediction.csv"
    if pred_csv.exists() and not force:
        return pred_csv
    if workspace.exists() and force:
        shutil.rmtree(workspace)
    cmd = ["deeprank-gnn-esm-predict", str(filtered_pdb), "A", "B", str(ncores)]
    result = subprocess.run(cmd, cwd=run_root)
    if result.returncode != 0:
        raise RuntimeError(f"{target}: filtered DeepRank run failed with returncode={result.returncode}")
    if not pred_csv.exists():
        raise RuntimeError(f"{target}: missing filtered prediction CSV {pred_csv}")
    return pred_csv


def write_complete_prediction(target: str, scores: list[float], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "predicted_fnat"])
        writer.writeheader()
        for idx, score in enumerate(scores):
            writer.writerow({"pdb_id": f"{target}_model_{idx}", "predicted_fnat": score})


def fill_target(target: str, work_dir: Path, log_path: Path | None, ncores: int, bad_score: float, force: bool) -> None:
    source_pdb = work_dir / "twochain_pdbs" / f"{target}.pdb"
    if not source_pdb.exists():
        raise FileNotFoundError(source_pdb)

    models = split_models(source_pdb)
    bad_indices = parse_bad_indices(log_path, target)
    if not bad_indices:
        raise RuntimeError(f"{target}: no bad model indices found in log; refusing to guess")

    keep_indices = [idx for idx in range(len(models)) if idx not in bad_indices]
    if not keep_indices:
        scores = [bad_score] * len(models)
    else:
        filtered_pdb = work_dir / "fallback_inputs" / f"{target}_filtered.pdb"
        write_filtered_pdb(models, keep_indices, filtered_pdb)
        pred_csv = run_filtered(target, filtered_pdb, work_dir, ncores, force)
        local_scores = parse_prediction_csv(pred_csv)
        scores = [bad_score] * len(models)
        for local_idx, original_idx in enumerate(keep_indices):
            if local_idx in local_scores:
                scores[original_idx] = local_scores[local_idx]

    out_csv = work_dir / "runs" / f"{target}-gnn_esm_pred_A_B" / "GNN_esm_prediction.csv"
    write_complete_prediction(target, scores, out_csv)
    bad_path = out_csv.with_suffix(".fallback_bad_models.txt")
    bad_path.write_text("\n".join(str(i) for i in sorted(bad_indices)) + "\n")
    print(f"{target}: wrote complete prediction CSV with {len(models)} models; bad={sorted(bad_indices)}")


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    log_path = Path(args.log) if args.log else None
    targets = args.target or read_failed_targets(work_dir)
    if not targets:
        raise SystemExit("no failed targets supplied or found")
    for target in targets:
        fill_target(target, work_dir, log_path, args.ncores, args.bad_score, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Repair failed DeepRank-GNN-esm target predictions.

This script is intentionally conservative:
  * reuse scores already printed in the full-run log when DeepRank crashed only
    while parsing its own CSV output;
  * assign a very low score to model indices that DeepRank reports as invalid
    graph/no-contact models;
  * score remaining missing models in small chunks and merge them back to a
    complete per-target GNN_esm_prediction.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path


BAD_SCORE = -1.0e9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--log", required=True, help="Full run log")
    parser.add_argument("--failed_file", default=None, help="Optional failed_targets.txt; unioned with log failures")
    parser.add_argument("--target", action="append", default=None)
    parser.add_argument("--ncores", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=20)
    parser.add_argument("--chunk_timeout", type=int, default=900, help="Seconds allowed for one DeepRank repair chunk")
    parser.add_argument("--bad_score", type=float, default=BAD_SCORE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
    if not lines:
        raise ValueError(f"no models in {pdb_path}")
    return [["MODEL        1\n", *lines, "ENDMDL\n"]]


def write_chunk_pdb(models: list[list[str]], indices: list[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        for serial, original_idx in enumerate(indices, start=1):
            lines = models[original_idx]
            handle.write(f"MODEL     {serial:4d}\n")
            if lines and lines[0].startswith("MODEL"):
                handle.writelines(lines[1:])
            else:
                handle.writelines(lines)
                handle.write("ENDMDL\n")
        handle.write("END\n")


def failed_targets_from_log(log_text: str) -> list[str]:
    return sorted(set(re.findall(r"\] (S-T\d+\.\d+): DeepRank failed", log_text)))


def target_bad_indices(log_text: str, target: str) -> set[int]:
    escaped = re.escape(target)
    bad = set(int(m.group(1)) for m in re.finditer(rf"{escaped}_model_(\d+)\.pkl", log_text))
    bad.update(int(m.group(1)) for m in re.finditer(rf"deleting {escaped}_model_(\d+)", log_text))
    return bad


def parse_predicted_lines(text: str, name_prefix: str | None = None) -> dict[int, float]:
    if name_prefix:
        pattern = rf"Predicted fnat for {re.escape(name_prefix)}_model_(\d+)\b.*?:\s*([-+]?\d+(?:\.\d+)?)"
    else:
        pattern = r"Predicted fnat for .*?_model_(\d+)\b.*?:\s*([-+]?\d+(?:\.\d+)?)"
    return {int(m.group(1)): float(m.group(2)) for m in re.finditer(pattern, text)}


def parse_prediction_csv(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return out
        if "pdb_id" not in reader.fieldnames or "predicted_fnat" not in reader.fieldnames:
            return out
        for row in reader:
            match = re.search(r"_model_(\d+)$", row.get("pdb_id", ""))
            if not match:
                continue
            try:
                out[int(match.group(1))] = float(row["predicted_fnat"])
            except (TypeError, ValueError):
                continue
    return out


def chunk_bad_local_indices(text: str, chunk_stem: str) -> set[int]:
    escaped = re.escape(chunk_stem)
    bad = set(int(m.group(1)) for m in re.finditer(rf"{escaped}_model_(\d+)\.pkl", text))
    bad.update(int(m.group(1)) for m in re.finditer(rf"deleting {escaped}_model_(\d+)", text))
    return bad


def run_chunk(
    target: str,
    chunk_id: int,
    indices: list[int],
    models: list[list[str]],
    work_dir: Path,
    ncores: int,
    force: bool,
    timeout: int,
) -> tuple[dict[int, float], set[int]]:
    chunk_stem = f"{target}_repair_chunk_{chunk_id:04d}"
    input_pdb = work_dir / "repair_inputs" / f"{chunk_stem}.pdb"
    run_root = work_dir / "repair_runs"
    workspace = run_root / f"{chunk_stem}-gnn_esm_pred_A_B"
    log_path = work_dir / "repair_logs" / f"{chunk_stem}.log"
    pred_csv = workspace / "GNN_esm_prediction.csv"

    if pred_csv.exists() and not force:
        return parse_prediction_csv(pred_csv), set()

    if workspace.exists() and force:
        shutil.rmtree(workspace)
    run_root.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_chunk_pdb(models, indices, input_pdb)

    cmd = ["deeprank-gnn-esm-predict", str(input_pdb), "A", "B", str(ncores)]
    try:
        result = subprocess.run(
            cmd,
            cwd=run_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = result.stdout or ""
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        output += f"\nTIMEOUT after {timeout}s for {chunk_stem} indices={indices}\n"
        log_path.write_text(output)
        return {}, set(range(len(indices)))
    log_path.write_text(output)

    scores = parse_prediction_csv(pred_csv)
    if not scores:
        scores = parse_predicted_lines(output, chunk_stem)
    bad_local = chunk_bad_local_indices(output, chunk_stem)
    return scores, bad_local


def write_complete_prediction(target: str, scores: list[float], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pdb_id", "predicted_fnat"])
        writer.writeheader()
        for idx, score in enumerate(scores):
            writer.writerow({"pdb_id": f"{target}_model_{idx}", "predicted_fnat": score})


def repair_target(
    target: str,
    work_dir: Path,
    log_text: str,
    ncores: int,
    chunk_size: int,
    chunk_timeout: int,
    bad_score: float,
    force: bool,
) -> None:
    out_csv = work_dir / "runs" / f"{target}-gnn_esm_pred_A_B" / "GNN_esm_prediction.csv"
    existing = parse_prediction_csv(out_csv)

    source_pdb = work_dir / "twochain_pdbs" / f"{target}.pdb"
    models = split_models(source_pdb)
    model_count = len(models)
    scores: list[float | None] = [None] * model_count

    for idx, score in existing.items():
        if idx < model_count:
            scores[idx] = score
    for idx, score in parse_predicted_lines(log_text, target).items():
        if idx < model_count and scores[idx] is None:
            scores[idx] = score

    bad_indices = target_bad_indices(log_text, target)
    for idx in bad_indices:
        if idx < model_count and scores[idx] is None:
            scores[idx] = bad_score

    def score_group(indices: list[int]) -> None:
        if not indices:
            return
        chunk_id = score_group.counter
        score_group.counter += 1
        local_scores, bad_local = run_chunk(target, chunk_id, indices, models, work_dir, ncores, force, chunk_timeout)
        for local_idx, score in local_scores.items():
            if local_idx < len(indices):
                scores[indices[local_idx]] = score
        for local_idx in bad_local:
            if local_idx < len(indices):
                scores[indices[local_idx]] = bad_score
        remaining = [idx for idx in indices if scores[idx] is None]
        if not remaining:
            return
        if len(indices) == 1:
            scores[indices[0]] = bad_score
            return
        midpoint = max(1, len(remaining) // 2)
        score_group(remaining[:midpoint])
        score_group(remaining[midpoint:])

    score_group.counter = 0  # type: ignore[attr-defined]

    missing = [idx for idx, score in enumerate(scores) if score is None]
    if bad_indices and len(missing) > chunk_size:
        # If the full target failed only because a few invalid/no-contact models
        # broke DeepRank's graph generation, a single filtered pass is much
        # faster than scoring many small chunks.
        score_group(missing)
        missing = [idx for idx, score in enumerate(scores) if score is None]
    for start in range(0, len(missing), chunk_size):
        score_group(missing[start : start + chunk_size])

    final_scores = [bad_score if score is None else float(score) for score in scores]
    write_complete_prediction(target, final_scores, out_csv)
    note = out_csv.with_suffix(".repair.txt")
    note.write_text(
        f"target={target}\nmodel_count={model_count}\n"
        f"bad_indices={','.join(map(str, sorted(bad_indices)))}\n"
        f"bad_score={bad_score}\n"
    )
    n_bad = sum(1 for score in final_scores if score == bad_score)
    print(f"{target}: repaired {model_count} models; bad_score_rows={n_bad}; out={out_csv}")


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    log_text = Path(args.log).read_text(errors="ignore")
    targets = list(args.target or [])
    if not targets:
        targets.extend(failed_targets_from_log(log_text))
        if args.failed_file:
            failed_file = Path(args.failed_file)
            if failed_file.exists():
                targets.extend(line.strip() for line in failed_file.read_text().splitlines() if line.strip())
    targets = sorted(dict.fromkeys(targets))
    if not targets:
        raise SystemExit("no failed targets found")
    for target in targets:
        repair_target(
            target,
            work_dir,
            log_text,
            args.ncores,
            args.chunk_size,
            args.chunk_timeout,
            args.bad_score,
            args.force,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

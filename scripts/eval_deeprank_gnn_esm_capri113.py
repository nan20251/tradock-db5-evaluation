#!/usr/bin/env python3
"""Run DeepRank-GNN-esm on CAPRI score-set targets.

The upstream CLI accepts exactly two chains. CAPRI score-set files can contain
multi-chain receptor/ligand groups, encoded in REMARK 3 as:

    RECEPTOR A LIGAND H L THETA ...

This script rewrites every MODEL into a two-chain representation before calling
DeepRank: receptor group -> chain A, ligand group -> chain B. Residues are
renumbered per MODEL and per new chain so Bio.PDB can parse merged groups.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


TOPKS = (1, 2, 5, 10, 20, 100)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", required=True, help="Directory with S-T*.pdb and S-T*.csv")
    parser.add_argument("--work_dir", required=True, help="Working directory for converted PDBs and DeepRank workspaces")
    parser.add_argument("--out", required=True, help="Per-decoy merged output CSV")
    parser.add_argument("--ncores", type=int, default=8, help="Cores passed to deeprank-gnn-esm-predict")
    parser.add_argument("--max_targets", type=int, default=None, help="Only process the first N targets")
    parser.add_argument("--only_target", action="append", default=None, help="Process only this target; can be repeated")
    parser.add_argument("--force_prepare", action="store_true", help="Regenerate converted two-chain PDBs")
    parser.add_argument("--force_run", action="store_true", help="Rerun DeepRank even when prediction CSV exists")
    parser.add_argument("--skip_run", action="store_true", help="Only prepare/merge/summarize existing predictions")
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def to_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def parse_remark_groups(pdb_path: Path) -> tuple[list[str], list[str]]:
    with pdb_path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("REMARK   3"):
                continue
            tokens = line.split()
            if "RECEPTOR" not in tokens or "LIGAND" not in tokens:
                continue
            receptor_idx = tokens.index("RECEPTOR")
            ligand_idx = tokens.index("LIGAND")
            try:
                theta_idx = tokens.index("THETA")
            except ValueError:
                theta_idx = len(tokens)
            receptor = tokens[receptor_idx + 1 : ligand_idx]
            ligand = tokens[ligand_idx + 1 : theta_idx]
            if receptor and ligand:
                return receptor, ligand

    chains: list[str] = []
    with pdb_path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
                chain = line[21].strip()
                if chain and chain not in chains:
                    chains.append(chain)
                    if len(chains) == 2:
                        return [chains[0]], [chains[1]]
    raise ValueError(f"cannot determine receptor/ligand chains for {pdb_path}")


def parse_model_serials(pdb_path: Path) -> list[str]:
    serials: list[str] = []
    with pdb_path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith("MODEL"):
                parts = line.split()
                serials.append(parts[1] if len(parts) > 1 else str(len(serials) + 1))
    return serials


def rewrite_two_chain_pdb(src: Path, dst: Path, receptor: list[str], ligand: list[str]) -> dict[str, object]:
    receptor_set = set(receptor)
    ligand_set = set(ligand)
    dst.parent.mkdir(parents=True, exist_ok=True)

    model_count = 0
    atom_count = 0
    skipped_atom_count = 0
    residue_maps: dict[str, dict[tuple[str, str, str], int]] = {"A": {}, "B": {}}
    residue_next = {"A": 1, "B": 1}

    def reset_residue_maps() -> None:
        residue_maps["A"].clear()
        residue_maps["B"].clear()
        residue_next["A"] = 1
        residue_next["B"] = 1

    with src.open(errors="ignore") as inp, dst.open("w") as out:
        reset_residue_maps()
        for raw in inp:
            if raw.startswith("MODEL"):
                model_count += 1
                reset_residue_maps()
                out.write(raw)
                continue
            if raw.startswith("ENDMDL"):
                out.write(raw)
                continue
            if raw.startswith("END"):
                out.write(raw)
                continue
            if not raw.startswith(("ATOM  ", "HETATM")):
                continue
            if len(raw) <= 26:
                skipped_atom_count += 1
                continue

            original_chain = raw[21].strip()
            if original_chain in receptor_set:
                new_chain = "A"
            elif original_chain in ligand_set:
                new_chain = "B"
            else:
                skipped_atom_count += 1
                continue

            residue_key = (original_chain, raw[22:26], raw[26])
            chain_map = residue_maps[new_chain]
            if residue_key not in chain_map:
                chain_map[residue_key] = residue_next[new_chain]
                residue_next[new_chain] += 1
            new_resseq = chain_map[residue_key]
            line = raw.rstrip("\n")
            out.write(line[:21] + new_chain + f"{new_resseq:4d}" + " " + line[27:] + "\n")
            atom_count += 1

        if model_count == 0:
            out.write("END\n")

    return {
        "target": src.stem,
        "receptor_chains": " ".join(receptor),
        "ligand_chains": " ".join(ligand),
        "model_count": model_count,
        "atom_count": atom_count,
        "skipped_atom_count": skipped_atom_count,
        "converted_pdb": str(dst),
    }


def load_labels(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def prediction_csv_for(work_dir: Path, target: str) -> Path:
    return work_dir / "runs" / f"{target}-gnn_esm_pred_A_B" / "GNN_esm_prediction.csv"


def run_deeprank(target: str, pdb_path: Path, work_dir: Path, ncores: int, force: bool) -> bool:
    pred_csv = prediction_csv_for(work_dir, target)
    if pred_csv.exists() and not force:
        log(f"{target}: prediction exists, skip")
        return True

    run_dir = work_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace = run_dir / f"{target}-gnn_esm_pred_A_B"
    if workspace.exists() and force:
        shutil.rmtree(workspace)

    cmd = ["deeprank-gnn-esm-predict", str(pdb_path), "A", "B", str(ncores)]
    log(f"{target}: run {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=run_dir)
    if result.returncode != 0:
        log(f"{target}: DeepRank failed with returncode={result.returncode}")
        return False
    if not pred_csv.exists():
        log(f"{target}: DeepRank finished but prediction CSV is missing")
        return False
    return True


def parse_predictions(pred_csv: Path) -> list[tuple[int, float, str]]:
    rows: list[tuple[int, float, str]] = []
    with pred_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = row.get("pdb_id", "")
            match = re.search(r"_model_(\d+)$", pdb_id)
            if not match:
                continue
            model_idx = int(match.group(1))
            rows.append((model_idx, to_float(row.get("predicted_fnat")), pdb_id))
    return rows


def merge_outputs(data_dir: Path, work_dir: Path, targets: list[Path], out_path: Path) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for target_idx, pdb_path in enumerate(targets, start=1):
        target = pdb_path.stem
        pred_csv = prediction_csv_for(work_dir, target)
        if not pred_csv.exists():
            continue
        labels = load_labels(data_dir / f"{target}.csv")
        model_serials = parse_model_serials(data_dir / f"{target}.pdb")
        for model_idx, score, pdb_id in parse_predictions(pred_csv):
            if model_idx >= len(labels):
                continue
            label = labels[model_idx]
            merged.append(
                {
                    "target_index": target_idx,
                    "target_total": len(targets),
                    "target": target,
                    "model_order": model_idx + 1,
                    "pdb_model_id": model_serials[model_idx] if model_idx < len(model_serials) else "",
                    "csv_model": label.get("model", ""),
                    "pdb_id": pdb_id,
                    "score": score,
                    "predicted_fnat": score,
                    "dockq": to_float(label.get("dockq")),
                    "fnat": to_float(label.get("fnat")),
                    "lrms": to_float(label.get("lrms")),
                    "irms": to_float(label.get("irms")),
                    "classification": label.get("classification", ""),
                    "identification": label.get("identification", ""),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "target_index",
        "target_total",
        "target",
        "model_order",
        "pdb_model_id",
        "csv_model",
        "pdb_id",
        "score",
        "predicted_fnat",
        "dockq",
        "fnat",
        "lrms",
        "irms",
        "classification",
        "identification",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)
    return merged


def is_positive(row: dict[str, object], metric: str) -> bool:
    if metric == "fnat03":
        return to_float(row.get("fnat")) > 0.3
    if metric == "dockq023":
        return to_float(row.get("dockq")) >= 0.23
    raise ValueError(metric)


def summarize(merged: list[dict[str, object]], out_path: Path) -> None:
    by_target: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in merged:
        by_target[str(row["target"])].append(row)

    summary_rows: list[dict[str, object]] = []
    for target, rows in sorted(by_target.items()):
        ranked = sorted(rows, key=lambda r: to_float(r.get("score"), -math.inf), reverse=True)
        top1 = ranked[0]
        summary: dict[str, object] = {
            "target": target,
            "n_models": len(rows),
            "top1_score": top1.get("score", ""),
            "top1_model_order": top1.get("model_order", ""),
            "top1_pdb_model_id": top1.get("pdb_model_id", ""),
            "top1_identification": top1.get("identification", ""),
            "top1_fnat": top1.get("fnat", ""),
            "top1_dockq": top1.get("dockq", ""),
            "best_fnat": max(to_float(r.get("fnat"), -math.inf) for r in rows),
            "best_dockq": max(to_float(r.get("dockq"), -math.inf) for r in rows),
            "n_fnat03": sum(is_positive(r, "fnat03") for r in rows),
            "n_dockq023": sum(is_positive(r, "dockq023") for r in rows),
        }
        for metric in ("fnat03", "dockq023"):
            for k in TOPKS:
                head = ranked[: min(k, len(ranked))]
                summary[f"success_{metric}@{k}"] = int(any(is_positive(r, metric) for r in head))
        summary_rows.append(summary)

    summary_path = out_path.with_suffix(".summary.csv")
    fields = list(summary_rows[0].keys()) if summary_rows else ["target"]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    aggregate_rows: list[dict[str, object]] = []
    for metric in ("fnat03", "dockq023"):
        for denom_name, rows_for_denom in (
            ("all_targets", summary_rows),
            (
                "positive_targets",
                [r for r in summary_rows if int(r[f"n_{metric}"]) > 0],
            ),
        ):
            denom = len(rows_for_denom)
            for k in TOPKS:
                key = f"success_{metric}@{k}"
                num = sum(int(r[key]) for r in rows_for_denom)
                aggregate_rows.append(
                    {
                        "metric": key,
                        "denominator_set": denom_name,
                        "numerator": num,
                        "denominator": denom,
                        "rate": "" if denom == 0 else num / denom,
                    }
                )

    aggregate_path = out_path.with_suffix(".summary.aggregate.csv")
    with aggregate_path.open("w", newline="") as handle:
        fields = ["metric", "denominator_set", "numerator", "denominator", "rate"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    log(f"merged rows -> {out_path}")
    log(f"per-target summary -> {summary_path}")
    log(f"aggregate summary -> {aggregate_path}")


def selected_targets(data_dir: Path, only_target: list[str] | None, max_targets: int | None) -> list[Path]:
    targets = sorted(data_dir.glob("S-T*.pdb"))
    if only_target:
        wanted = set(only_target)
        targets = [p for p in targets if p.stem in wanted]
    if max_targets is not None:
        targets = targets[:max_targets]
    return targets


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    work_dir = Path(args.work_dir)
    out_path = Path(args.out)
    prepared_dir = work_dir / "twochain_pdbs"
    meta_path = work_dir / "target_chain_groups.csv"

    targets = selected_targets(data_dir, args.only_target, args.max_targets)
    if not targets:
        raise SystemExit("no targets selected")

    meta_rows: list[dict[str, object]] = []
    for pdb_path in targets:
        target = pdb_path.stem
        converted = prepared_dir / f"{target}.pdb"
        receptor, ligand = parse_remark_groups(pdb_path)
        if args.force_prepare or not converted.exists():
            row = rewrite_two_chain_pdb(pdb_path, converted, receptor, ligand)
        else:
            row = {
                "target": target,
                "receptor_chains": " ".join(receptor),
                "ligand_chains": " ".join(ligand),
                "model_count": "",
                "atom_count": "",
                "skipped_atom_count": "",
                "converted_pdb": str(converted),
            }
        meta_rows.append(row)

    work_dir.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", newline="") as handle:
        fields = ["target", "receptor_chains", "ligand_chains", "model_count", "atom_count", "skipped_atom_count", "converted_pdb"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(meta_rows)
    log(f"chain metadata -> {meta_path}")

    if shutil.which("deeprank-gnn-esm-predict") is None and not args.skip_run:
        raise SystemExit("deeprank-gnn-esm-predict not found in PATH")

    failures: list[str] = []
    if not args.skip_run:
        for idx, pdb_path in enumerate(targets, start=1):
            target = pdb_path.stem
            log(f"target {idx}/{len(targets)} {target}")
            converted = prepared_dir / f"{target}.pdb"
            ok = run_deeprank(target, converted, work_dir, args.ncores, args.force_run)
            if not ok:
                failures.append(target)
            merged = merge_outputs(data_dir, work_dir, targets, out_path)
            if merged:
                summarize(merged, out_path)

    merged = merge_outputs(data_dir, work_dir, targets, out_path)
    if merged:
        summarize(merged, out_path)
    else:
        log("no predictions merged yet")

    if failures:
        failure_path = work_dir / "failed_targets.txt"
        failure_path.write_text("\n".join(failures) + "\n")
        log(f"failed targets -> {failure_path}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Create a within-target structure montage for different ranking criteria."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


SVG_STYLE = """
text{font-family:Arial,Helvetica,sans-serif}
.title{font-size:24px;font-weight:700}
.subtitle{font-size:13px;fill:#444}
.row-label{font-size:14px;font-weight:700}
.panel-title{font-size:12px;font-weight:700}
.metric{font-size:10px;fill:#333}
.note{font-size:11px;fill:#555}
"""


GROUP_SPECS = [
    ("tradock_mdn_top5", "TraDock/MDN top 5", "mdn_score", True, "#20854d"),
    ("tradock_mdn_bottom5", "TraDock/MDN bottom 5", "mdn_score", False, "#b94436"),
    ("original_first5", "Original first 5", "original_index", False, "#2f6fb0"),
    ("original_last5", "Original last 5", "original_index", True, "#8a5fbf"),
    ("fnat_top5", "Fnat top 5", "fnat", True, "#007b83"),
    ("fnat_bottom5", "Fnat bottom 5", "fnat", False, "#b05b00"),
    ("dockq_top5", "DockQ top 5", "dockq", True, "#7a6f00"),
    ("dockq_bottom5", "DockQ bottom 5", "dockq", False, "#9a4b76"),
]


def to_float(value, default=0.0):
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def to_int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def split_models(pdb_path):
    models = {}
    current_model = None
    current_chains = {}
    with open(pdb_path, errors="ignore") as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec == "MODEL":
                current_model = int(line.split()[1])
                current_chains = {}
            elif rec == "ENDMDL":
                if current_model is not None:
                    models[current_model] = current_chains
                current_model = None
            elif rec in ("ATOM", "HETATM", "TER", "ANISOU") and current_model is not None:
                if len(line) > 21:
                    current_chains.setdefault(line[21], []).append(line)

    if models:
        return models

    chains = {}
    with open(pdb_path, errors="ignore") as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec in ("ATOM", "HETATM", "TER", "ANISOU") and len(line) > 21:
                chains.setdefault(line[21], []).append(line)
    return {0: chains} if chains else {}


def pick_rec_lig_chains(chains):
    chain_ids = list(chains.keys())
    if len(chain_ids) < 2:
        return None, None
    rec_ch = "A" if "A" in chains else chain_ids[0]
    lig_ch = "B" if "B" in chains else chain_ids[1]
    return rec_ch, lig_ch


def atom_key(line):
    return (
        line[21],
        line[22:26].strip(),
        line[26].strip(),
        line[12:16].strip(),
    )


def atom_xyz(line):
    return np.array(
        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
        dtype=np.float64,
    )


def common_atom_arrays(ref_lines, mob_lines, ca_only=True):
    def build(lines):
        out = {}
        for line in lines:
            if line[:6].strip() not in ("ATOM", "HETATM") or len(line) < 54:
                continue
            if ca_only and line[12:16].strip() != "CA":
                continue
            try:
                out[atom_key(line)] = atom_xyz(line)
            except ValueError:
                continue
        return out

    ref = build(ref_lines)
    mob = build(mob_lines)
    keys = sorted(set(ref) & set(mob))
    if len(keys) < 3 and ca_only:
        return common_atom_arrays(ref_lines, mob_lines, ca_only=False)
    if len(keys) < 3:
        return None, None
    return np.vstack([ref[k] for k in keys]), np.vstack([mob[k] for k in keys])


def kabsch_to_reference(ref_points, mob_points):
    ref_center = ref_points.mean(axis=0)
    mob_center = mob_points.mean(axis=0)
    ref0 = ref_points - ref_center
    mob0 = mob_points - mob_center
    h = mob0.T @ ref0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    return r, mob_center, ref_center


def transform_xyz(xyz, transform):
    r, mob_center, ref_center = transform
    return (xyz - mob_center) @ r + ref_center


def update_pdb_line_xyz(line, xyz):
    return f"{line[:30]}{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{line[54:]}"


def transform_chains(chains, transform):
    out = {}
    for chain_id, lines in chains.items():
        new_lines = []
        for line in lines:
            if line[:6].strip() in ("ATOM", "HETATM", "ANISOU") and len(line) >= 54:
                if line[:6].strip() == "ANISOU":
                    new_lines.append(line)
                    continue
                try:
                    new_lines.append(update_pdb_line_xyz(line, transform_xyz(atom_xyz(line), transform)))
                except ValueError:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        out[chain_id] = new_lines
    return out


def parse_draw_points(lines):
    ca = []
    atoms = []
    for line in lines:
        if line[:6].strip() not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        try:
            xyz = atom_xyz(line)
        except ValueError:
            continue
        atoms.append(xyz)
        if line[12:16].strip() == "CA":
            ca.append(xyz)
    pts = ca if ca else atoms
    return np.asarray(pts, dtype=np.float64)


def write_model_pdb(chains, out_path, model_id):
    with open(out_path, "w") as handle:
        handle.write(f"MODEL {int(model_id):8d}\n")
        for chain_id in chains:
            for line in chains[chain_id]:
                rec = line[:6].strip()
                if rec in ("ATOM", "HETATM", "TER", "ANISOU"):
                    handle.write(line)
        handle.write("ENDMDL\n")


def read_result_rows(result_csv, target, original_index_by_model):
    rows = []
    with open(result_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("target") != target:
                continue
            mid = to_int(row.get("model_id"))
            rows.append(
                {
                    "target": target,
                    "model_id": mid,
                    "original_index": original_index_by_model.get(mid, mid),
                    "identification": row.get("identification", ""),
                    "mdn_score": to_float(row.get("mdn_score", row.get("score"))),
                    "fnat": to_float(row.get("fnat")),
                    "dockq": to_float(row.get("dockq")),
                    "lrms": to_float(row.get("lrms")),
                    "irms": to_float(row.get("irms")),
                    "classification": row.get("classification", ""),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows for target {target} in {result_csv}")
    return rows


def select_rows(rows):
    selected = []
    seen = set()
    for group_key, group_label, metric, descending, color in GROUP_SPECS:
        ranked = sorted(rows, key=lambda row: row[metric], reverse=descending)
        group_rows = ranked[:5]
        for rank, row in enumerate(group_rows, 1):
            item = dict(row)
            item.update(
                {
                    "group_key": group_key,
                    "group_label": group_label,
                    "rank_in_group": rank,
                    "sort_metric": metric,
                    "sort_descending": descending,
                    "color": color,
                }
            )
            selected.append(item)
            seen.add(item["model_id"])
    return selected, seen


def sample_points(points, max_points):
    if len(points) <= max_points:
        return points
    idx = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[idx]


def pca_basis(points):
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return center, vh[:2].T


def project(points, center, basis):
    return (points - center) @ basis


def fit_projected(xy, x0, y0, width, height, global_min, global_max, pad=18):
    span = np.maximum(global_max - global_min, 1e-6)
    scale = min((width - 2 * pad) / span[0], (height - 2 * pad) / span[1])
    out = (xy - global_min) * scale
    used = span * scale
    out[:, 0] += x0 + (width - used[0]) / 2
    out[:, 1] = y0 + height - (out[:, 1] + (height - used[1]) / 2)
    return out


def draw_panel(svg, item, x, y, w, h, center, basis, global_min, global_max):
    stroke = "#222" if item["group_key"] == "tradock_mdn_top5" and item["rank_in_group"] == 1 else "#d6d6d6"
    stroke_width = "2.2" if stroke == "#222" else "1"
    svg.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="3" fill="#fbfbfb" '
        f'stroke="{stroke}" stroke-width="{stroke_width}"/>'
    )
    badge = " Top1 success" if item["group_key"] == "tradock_mdn_top5" and item["rank_in_group"] == 1 else ""
    svg.append(
        f'<text x="{x+9}" y="{y+17}" class="panel-title">'
        f'#{item["rank_in_group"]} model {item["model_id"]}{svg_escape(badge)}</text>'
    )
    svg.append(
        f'<text x="{x+9}" y="{y+33}" class="metric">'
        f'{svg_escape(item["identification"])}  orig#{item["original_index"]}</text>'
    )
    svg.append(
        f'<text x="{x+9}" y="{y+48}" class="metric">'
        f'MDN={item["mdn_score"]:.4f}  Fnat={item["fnat"]:.3f}  DockQ={item["dockq"]:.3f}</text>'
    )
    svg.append(
        f'<text x="{x+9}" y="{y+63}" class="metric">'
        f'LRMS={item["lrms"]:.2f}  IRMS={item["irms"]:.2f}  {svg_escape(item["classification"])}</text>'
    )

    rec = sample_points(item["rec_points"], 500)
    lig = sample_points(item["lig_points"], 400)
    if len(rec) == 0 or len(lig) == 0:
        svg.append(f'<text x="{x+9}" y="{y+92}" class="metric">No drawable atoms</text>')
        return
    rec_xy = fit_projected(project(rec, center, basis), x + 5, y + 72, w - 10, h - 78, global_min, global_max)
    lig_xy = fit_projected(project(lig, center, basis), x + 5, y + 72, w - 10, h - 78, global_min, global_max)
    for px, py in rec_xy:
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.05" fill="#8d8d8d" opacity="0.36"/>')
    for px, py in lig_xy:
        svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.75" fill="{item["color"]}" opacity="0.82"/>')


def make_montage(target, selected, out_svg):
    groups = []
    for spec in GROUP_SPECS:
        group_key = spec[0]
        groups.append([item for item in selected if item["group_key"] == group_key])

    all_draw = []
    for item in selected:
        all_draw.append(sample_points(item["rec_points"], 500))
        all_draw.append(sample_points(item["lig_points"], 400))
    center, basis = pca_basis(np.vstack(all_draw))
    projected = np.vstack([project(points, center, basis) for points in all_draw if len(points)])
    global_min = projected.min(axis=0)
    global_max = projected.max(axis=0)

    width = 1960
    panel_w = 340
    panel_h = 220
    label_w = 170
    margin_x = 32
    gap = 12
    row_gap = 18
    top_y = 112
    height = top_y + len(groups) * panel_h + (len(groups) - 1) * row_gap + 36

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<style>{SVG_STYLE}</style>',
        f'<text x="32" y="34" class="title">{svg_escape(target)} single-target structure comparison</text>',
        '<text x="32" y="58" class="subtitle">Each panel is one decoy from the same target. Gray = receptor chain A; colored = ligand chain B. All structures are aligned by receptor A and drawn with the same projection/scale.</text>',
        '<text x="32" y="78" class="subtitle">Fnat/DockQ are labels computed against native; the native PDB itself is not included in this CAPRI decoy file.</text>',
        '<text x="32" y="98" class="note">Black border marks the TraDock Top1-success model selected by MDN score.</text>',
    ]

    for row_idx, group_items in enumerate(groups):
        if not group_items:
            continue
        y = top_y + row_idx * (panel_h + row_gap)
        svg.append(
            f'<text x="{margin_x}" y="{y+21}" class="row-label" fill="{group_items[0]["color"]}">'
            f'{svg_escape(group_items[0]["group_label"])}</text>'
        )
        x0 = margin_x + label_w
        for col_idx, item in enumerate(group_items):
            x = x0 + col_idx * (panel_w + gap)
            draw_panel(svg, item, x, y, panel_w, panel_h, center, basis, global_min, global_max)

    svg.append("</svg>")
    out_svg.write_text("\n".join(svg))


def write_selection_csv(selected, out_csv):
    fields = [
        "group_key",
        "group_label",
        "rank_in_group",
        "sort_metric",
        "sort_descending",
        "target",
        "model_id",
        "original_index",
        "identification",
        "mdn_score",
        "fnat",
        "dockq",
        "lrms",
        "irms",
        "classification",
        "extracted_pdb",
    ]
    with open(out_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in selected:
            writer.writerow({key: item.get(key, "") for key in fields})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--result_csv", required=True)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    target = args.target
    out_dir = Path(args.out_dir)
    structures_dir = out_dir / "structures"
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir.mkdir(parents=True, exist_ok=True)

    models = split_models(args.pdb)
    if not models:
        raise RuntimeError(f"No models parsed from {args.pdb}")
    model_ids = sorted(models)
    original_index_by_model = {mid: idx + 1 for idx, mid in enumerate(model_ids)}
    rows = read_result_rows(args.result_csv, target, original_index_by_model)
    selected, needed_model_ids = select_rows(rows)

    top1 = next(item for item in selected if item["group_key"] == "tradock_mdn_top5" and item["rank_in_group"] == 1)
    ref_chains = models.get(top1["model_id"])
    if not ref_chains:
        raise RuntimeError(f"Top1 model {top1['model_id']} not found in {args.pdb}")
    ref_rec, _ = pick_rec_lig_chains(ref_chains)
    if ref_rec is None:
        raise RuntimeError("Reference model has fewer than two chains")

    aligned_by_model = {}
    for mid in needed_model_ids:
        chains = models.get(mid)
        if not chains:
            continue
        rec_ch, lig_ch = pick_rec_lig_chains(chains)
        if rec_ch is None:
            continue
        ref_points, mob_points = common_atom_arrays(ref_chains[ref_rec], chains[rec_ch], ca_only=True)
        if ref_points is None:
            continue
        aligned_by_model[mid] = transform_chains(chains, kabsch_to_reference(ref_points, mob_points))

    drawable = []
    for item in selected:
        mid = item["model_id"]
        chains = aligned_by_model.get(mid)
        if not chains:
            continue
        rec_ch, lig_ch = pick_rec_lig_chains(chains)
        if rec_ch is None:
            continue
        pdb_name = f'{item["group_key"]}_rank{item["rank_in_group"]}_{target}_model{mid}.pdb'
        pdb_out = structures_dir / pdb_name
        write_model_pdb(chains, pdb_out, mid)
        item = dict(item)
        item["rec_points"] = parse_draw_points(chains[rec_ch])
        item["lig_points"] = parse_draw_points(chains[lig_ch])
        item["extracted_pdb"] = str(pdb_out)
        drawable.append(item)

    expected = len(selected)
    if len(drawable) != expected:
        print(f"warning: drawable structures {len(drawable)} != selected rows {expected}")

    out_svg = out_dir / f"{target}_single_target_rank_montage.svg"
    out_csv = out_dir / f"{target}_single_target_rank_montage.csv"
    make_montage(target, drawable, out_svg)
    write_selection_csv(drawable, out_csv)
    print(f"svg -> {out_svg}")
    print(f"csv -> {out_csv}")
    print(f"structures -> {structures_dir}")


if __name__ == "__main__":
    main()

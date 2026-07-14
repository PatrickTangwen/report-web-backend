import argparse
import csv
import hashlib
import json
import secrets
from collections import Counter
from datetime import date
from pathlib import Path

from fibrotic_contract import MATCH_FIELDS, PUBLIC_FIELDS, TARGETS


PRESET_FEATURE_FIELDS = [
    field
    for field in MATCH_FIELDS
    if field not in {"visual_reference_id", "disease"}
]


def _read_source(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    required = (set(PUBLIC_FIELDS) - {"visual_reference_id"}) | (
        set(MATCH_FIELDS) - {"visual_reference_id"}
    )
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Source is missing required columns: {sorted(missing)}")
    counts = Counter(row["disease"] for row in rows)
    missing_targets = TARGETS - set(counts)
    if missing_targets:
        raise ValueError(f"Source is missing comparison targets: {sorted(missing_targets)}")
    return rows, counts


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _preset_rank(source_row):
    """Rank a fixed display preset without using display geometry or model output."""
    feature_signature = "\x1f".join(source_row[field] for field in PRESET_FEATURE_FIELDS)
    return hashlib.sha256(feature_signature.encode()).hexdigest()


def _select_preset(source_rows, display_rows):
    candidates = [
        (source_row, display_row)
        for source_row, display_row in zip(source_rows, display_rows)
        if source_row["disease"] == "Pulmonary_fibrosis"
    ]
    selected = sorted(candidates, key=lambda pair: _preset_rank(pair[0]))[:12]
    return [display_row for _, display_row in selected]


def build_release(source, public_dir, private_output, release_date):
    source_rows, disease_counts = _read_source(source)
    display_rows = []
    private_rows = []

    for source_row in source_rows:
        visual_id = "vr_" + secrets.token_urlsafe(18)
        joined = {**source_row, "visual_reference_id": visual_id}
        display_rows.append({field: joined[field] for field in PUBLIC_FIELDS})
        private_rows.append({field: joined[field] for field in MATCH_FIELDS})

    public_dir.mkdir(parents=True, exist_ok=True)
    display_path = public_dir / "fibrotic_embedding.csv"
    _write_csv(display_path, PUBLIC_FIELDS, display_rows)
    _write_csv(private_output, MATCH_FIELDS, private_rows)

    display_digest = hashlib.sha256(display_path.read_bytes()).hexdigest()
    private_digest = hashlib.sha256(private_output.read_bytes()).hexdigest()
    dataset_version = f"fibrotic-{release_date}-{display_digest[:12]}"
    manifest = {
        "dataset_version": dataset_version,
        "release_date": release_date,
        "point_count": len(display_rows),
        "disease_counts": dict(sorted(disease_counts.items())),
        "public_schema": PUBLIC_FIELDS,
        "display_sha256": display_digest,
        "private_schema": MATCH_FIELDS,
        "private_sha256": private_digest,
    }
    (public_dir / "fibrotic_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    selected = _select_preset(source_rows, display_rows)
    preset = {
        "dataset_version": dataset_version,
        "preset_id": "pulmonary-fibrosis-walkthrough",
        "label": "Pulmonary fibrosis display preset",
        "target": "Pulmonary_fibrosis",
        "display_mode": "overview",
        "visual_reference_ids": [row["visual_reference_id"] for row in selected],
        "summary": {
            "reference_count": len(selected),
            "title": "Pulmonary fibrosis display preset",
            "description": (
                "A fixed set of reference points selected independently of the "
                "t-SNE layout to demonstrate the visualization. It is not a "
                "clinical similarity result or personal assessment."
            ),
        },
    }
    (public_dir / "fibrotic_preset.json").write_text(
        json.dumps(preset, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--release-date", default=date.today().isoformat())
    args = parser.parse_args()
    build_release(
        args.source,
        args.public_dir,
        args.private_output,
        args.release_date,
    )


if __name__ == "__main__":
    main()

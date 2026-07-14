import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from fibrotic_contract import PUBLIC_FIELDS


@lru_cache(maxsize=1)
def load_fibrotic_release():
    release_dir = Path(__file__).parent / "data" / "fibrotic_release"
    display_path = release_dir / "fibrotic_embedding.csv"
    manifest = json.loads((release_dir / "fibrotic_manifest.json").read_text())
    preset = json.loads((release_dir / "fibrotic_preset.json").read_text())

    digest = hashlib.sha256(display_path.read_bytes()).hexdigest()
    if digest != manifest["display_sha256"]:
        raise RuntimeError("Fibrotic display artifact does not match its manifest")
    if manifest["public_schema"] != PUBLIC_FIELDS:
        raise RuntimeError("Fibrotic display schema is not approved")

    with display_path.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != PUBLIC_FIELDS:
            raise RuntimeError("Fibrotic display artifact has an unexpected schema")
        points = [
            {
                **row,
                "tsne_x": float(row["tsne_x"]),
                "tsne_y": float(row["tsne_y"]),
            }
            for row in reader
        ]

    if len(points) != manifest["point_count"]:
        raise RuntimeError("Fibrotic display point count does not match its manifest")
    point_ids = {point["visual_reference_id"] for point in points}
    if len(point_ids) != len(points):
        raise RuntimeError("Fibrotic Visual Reference IDs are not unique")
    if preset["dataset_version"] != manifest["dataset_version"]:
        raise RuntimeError("Fibrotic preset and display release versions differ")
    if preset.get("display_mode") not in {"compact", "multi_region", "overview"}:
        raise RuntimeError("Fibrotic preset has an unsupported display mode")
    if preset["display_mode"] == "multi_region" and not (
        isinstance(preset.get("minimum_region_size"), int)
        and preset["minimum_region_size"] >= 1
    ):
        raise RuntimeError("Multi-region preset is missing its geometry contract")
    if not set(preset["visual_reference_ids"]) <= point_ids:
        raise RuntimeError("Fibrotic preset references points outside the release")

    embedding = {
        "dataset_version": manifest["dataset_version"],
        "release_date": manifest["release_date"],
        "point_count": manifest["point_count"],
        "disease_counts": manifest["disease_counts"],
        "points": points,
    }
    return embedding, preset

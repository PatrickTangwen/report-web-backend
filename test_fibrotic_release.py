import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import app
from build_fibrotic_release import MATCH_FIELDS, _select_preset


TARGETS = [
    ("CKD", "CKD", "overlap"),
    ("Cardiac_Fibrosis", "Cardiac Fibrosis", "pure"),
    ("MASH", "MASH", "intermediate"),
    ("Pulmonary_fibrosis", "Pulmonary Fibrosis", "pure"),
    ("SSc_Connective_Tissue", "Systemic Sclerosis / Connective Tissue", "overlap"),
    ("Crohns_Disease", "Crohn's Disease", "intermediate"),
    ("Fibrosis_of_Skin", "Skin Fibrosis", "pure"),
]


@pytest_asyncio.fixture
async def api_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _write_source(path):
    fieldnames = [
        "eid",
        "disease",
        "group",
        "tsne_x",
        "tsne_y",
        "age_recruit",
        "sex",
        "BMI",
        "waist",
        "hip",
        "height",
        "weight",
        "DBP",
        "SBP",
        "creatinine",
        "HbA1c",
        "smoking_status",
        "alcohol_freq",
        "has_affected_rel",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, (disease, _, group) in enumerate(TARGETS, start=1):
            for offset in range(3):
                writer.writerow(
                    {
                        "eid": 9000000 + index * 10 + offset,
                        "disease": disease,
                        "group": group,
                        "tsne_x": index + offset / 10,
                        "tsne_y": -index - offset / 10,
                        "age_recruit": 40 + index,
                        "sex": index % 2,
                        "BMI": 20 + index,
                        "waist": 80 + index,
                        "hip": 90 + index,
                        "height": 160 + index,
                        "weight": 65 + index,
                        "DBP": 70 + index,
                        "SBP": 110 + index,
                        "creatinine": 60 + index,
                        "HbA1c": 30 + index,
                        "smoking_status": index % 3,
                        "alcohol_freq": index % 5,
                        "has_affected_rel": index % 2,
                    }
                )


def test_release_cli_builds_linked_display_and_private_artifacts(tmp_path):
    source = tmp_path / "source.csv"
    public_dir = tmp_path / "public"
    private_path = tmp_path / "private" / "fibrotic_match.csv"
    _write_source(source)

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("build_fibrotic_release.py")),
            "--source",
            str(source),
            "--public-dir",
            str(public_dir),
            "--private-output",
            str(private_path),
            "--release-date",
            "2026-07-13",
        ],
        check=True,
    )

    with (public_dir / "fibrotic_embedding.csv").open(newline="") as file:
        display_rows = list(csv.DictReader(file))
    with private_path.open(newline="") as file:
        private_rows = list(csv.DictReader(file))
    manifest = json.loads((public_dir / "fibrotic_manifest.json").read_text())
    preset = json.loads((public_dir / "fibrotic_preset.json").read_text())

    assert set(display_rows[0]) == {
        "visual_reference_id",
        "disease",
        "group",
        "tsne_x",
        "tsne_y",
    }
    assert "eid" not in private_rows[0]
    assert "id" not in private_rows[0]
    assert not {"tsne_x", "tsne_y", "group", "p_true", "p_max", "correct"} & set(
        private_rows[0]
    )

    display_ids = [row["visual_reference_id"] for row in display_rows]
    private_ids = [row["visual_reference_id"] for row in private_rows]
    assert len(display_ids) == len(set(display_ids)) == 21
    assert set(display_ids) == set(private_ids)
    assert all(len(value) >= 20 for value in display_ids)
    assert not set(display_ids) & {str(9000000 + i) for i in range(100)}

    assert manifest["dataset_version"].startswith("fibrotic-2026-07-13-")
    assert manifest["point_count"] == 21
    assert manifest["public_schema"] == list(display_rows[0])
    assert manifest["private_schema"] == list(private_rows[0])
    assert manifest["private_sha256"] == hashlib.sha256(private_path.read_bytes()).hexdigest()
    assert set(manifest["disease_counts"]) == {disease for disease, _, _ in TARGETS}

    assert preset["dataset_version"] == manifest["dataset_version"]
    assert preset["preset_id"] == "pulmonary-fibrosis-walkthrough"
    assert preset["target"] == "Pulmonary_fibrosis"
    assert preset["display_mode"] == "overview"
    assert set(preset["visual_reference_ids"]) <= set(display_ids)
    assert preset["summary"]["reference_count"] == len(preset["visual_reference_ids"])
    assert "risk" not in json.dumps(preset).lower()


def test_release_cli_generates_new_visual_ids_for_each_release(tmp_path):
    source = tmp_path / "source.csv"
    _write_source(source)
    id_sets = []

    for run in ("first", "second"):
        public_dir = tmp_path / run / "public"
        private_path = tmp_path / run / "private.csv"
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("build_fibrotic_release.py")),
                "--source",
                str(source),
                "--public-dir",
                str(public_dir),
                "--private-output",
                str(private_path),
                "--release-date",
                "2026-07-13",
            ],
            check=True,
        )
        with (public_dir / "fibrotic_embedding.csv").open(newline="") as file:
            id_sets.append({row["visual_reference_id"] for row in csv.DictReader(file)})

    assert id_sets[0].isdisjoint(id_sets[1])


def test_preset_selection_does_not_use_tsne_layout():
    source_rows = []
    display_rows = []
    for index in range(15):
        source_row = {
            field: str(index + field_index)
            for field_index, field in enumerate(MATCH_FIELDS)
            if field != "visual_reference_id"
        }
        source_row["disease"] = "Pulmonary_fibrosis"
        source_rows.append(source_row)
        display_rows.append(
            {
                "visual_reference_id": f"vr_{index}",
                "disease": "Pulmonary_fibrosis",
                "group": "pure",
                "tsne_x": str(index),
                "tsne_y": str(-index),
            }
        )

    initial = [
        row["visual_reference_id"]
        for row in _select_preset(source_rows, display_rows)
    ]
    for index, row in enumerate(display_rows):
        row["tsne_x"] = str(10000 - index * 100)
        row["tsne_y"] = str(index * index)
    changed_layout = [
        row["visual_reference_id"]
        for row in _select_preset(source_rows, display_rows)
    ]

    assert changed_layout == initial
    assert len(initial) == 12


@pytest.mark.asyncio
async def test_embedding_endpoint_serves_only_display_safe_release(api_client):
    response = await api_client.get("/embedding/fibrotic")

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_version"].startswith("fibrotic-2026-07-13-")
    assert body["point_count"] == len(body["points"]) == 6010
    assert set(body["points"][0]) == {
        "visual_reference_id",
        "disease",
        "group",
        "tsne_x",
        "tsne_y",
    }
    assert response.headers["etag"] == f'"{body["dataset_version"]}"'
    assert response.headers["cache-control"] == "public, max-age=300"

    cached = await api_client.get(
        "/embedding/fibrotic",
        headers={"If-None-Match": response.headers["etag"]},
    )
    assert cached.status_code == 304
    assert not cached.content


@pytest.mark.asyncio
async def test_preset_endpoint_references_the_active_release(api_client):
    embedding = (await api_client.get("/embedding/fibrotic")).json()
    response = await api_client.get("/embedding/fibrotic/preset")

    assert response.status_code == 200
    preset = response.json()
    assert preset["dataset_version"] == embedding["dataset_version"]
    assert preset["preset_id"] == "pulmonary-fibrosis-walkthrough"
    assert preset["target"] == "Pulmonary_fibrosis"
    assert preset["display_mode"] == "overview"
    point_ids = {point["visual_reference_id"] for point in embedding["points"]}
    assert set(preset["visual_reference_ids"]) <= point_ids
    assert preset["summary"]["reference_count"] == len(
        preset["visual_reference_ids"]
    )

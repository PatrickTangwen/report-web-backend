from copy import deepcopy
import csv
import hashlib
import json

import pytest_asyncio
import pytest
from httpx import ASGITransport, AsyncClient

from app import app, get_profile_matching_release
from demo_profile import build_profile_draft, confirm_profile
from calibrate_profile_matching import calibrate_matching
from fibrotic_contract import MATCH_FIELDS, PUBLIC_FIELDS
from profile_matching import (
    match_confirmed_profile,
    read_matching_release,
    resolve_private_matching_path,
)


def candidate(field, value, unit=None, source=None):
    return {
        "field": field,
        "raw_value": value,
        "raw_unit": unit,
        "source_text": source or f"{field} {value}",
        "operation": "set",
    }


def confirmed_profile(*candidates):
    return confirm_profile(build_profile_draft(list(candidates)))


def test_private_matching_path_uses_local_artifact_without_dataset_config(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("FIBROTIC_MATCH_DATASET_REPO", raising=False)

    assert resolve_private_matching_path(tmp_path) == (
        tmp_path / "private" / "fibrotic_match.csv"
    )


def test_private_matching_path_downloads_from_configured_private_dataset(
    tmp_path, monkeypatch
):
    downloaded = tmp_path / "cache" / "fibrotic_match.csv"
    calls = []
    monkeypatch.setenv("FIBROTIC_MATCH_DATASET_REPO", "patirckistc/report-web-private")
    monkeypatch.setenv("FIBROTIC_MATCH_DATASET_REVISION", "release-2026-07-13")
    monkeypatch.setenv("HF_TOKEN", "test-secret-token")

    def download_file(**kwargs):
        calls.append(kwargs)
        return str(downloaded)

    result = resolve_private_matching_path(tmp_path, download_file=download_file)

    assert result == downloaded
    assert calls == [
        {
            "repo_id": "patirckistc/report-web-private",
            "filename": "fibrotic_match.csv",
            "repo_type": "dataset",
            "revision": "release-2026-07-13",
            "token": "test-secret-token",
        }
    ]


def test_private_matching_path_requires_token_for_private_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("FIBROTIC_MATCH_DATASET_REPO", "patirckistc/report-web-private")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN secret is required"):
        resolve_private_matching_path(tmp_path, download_file=lambda **_: None)


def test_private_matching_download_error_does_not_expose_token(tmp_path, monkeypatch):
    secret = "test-secret-token"
    monkeypatch.setenv("FIBROTIC_MATCH_DATASET_REPO", "patirckistc/report-web-private")
    monkeypatch.setenv("HF_TOKEN", secret)

    def fail_download(**_):
        raise OSError(f"authentication failed for {secret}")

    with pytest.raises(RuntimeError) as error:
        resolve_private_matching_path(tmp_path, download_file=fail_download)

    assert str(error.value) == "Private matching dataset could not be downloaded"
    assert secret not in str(error.value)


@pytest_asyncio.fixture
async def api_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


def test_matching_is_target_scoped_and_ignores_visualization_and_outcome_fields():
    profile = confirmed_profile(
        candidate("age", 55, source="age 55"),
        candidate("sex", "female", source="sex female"),
    )
    rows = []
    for index in range(5):
        rows.append(
            {
                "visual_reference_id": f"vr_ckd_{index}",
                "disease": "CKD",
                "age_recruit": "55",
                "sex": "0",
                "tsne_x": str(index * 1000),
                "tsne_y": str(index * -1000),
                "p_true": str(index / 10),
            }
        )
        rows.append(
            {
                "visual_reference_id": f"vr_mash_{index}",
                "disease": "MASH",
                "age_recruit": "55",
                "sex": "0",
                "tsne_x": "0",
                "tsne_y": "0",
                "p_true": "1",
            }
        )
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "CKD": {
                "eligible_patterns": {
                    "demographics": {"distance_threshold": 0.01}
                }
            }
        },
    }

    first = match_confirmed_profile(
        profile,
        "CKD",
        rows,
        calibration,
        "fibrotic-test-release",
    )
    changed_prohibited_fields = deepcopy(rows)
    for row in changed_prohibited_fields:
        row["tsne_x"] = "999999"
        row["tsne_y"] = "-999999"
        row["p_true"] = "0" if row["p_true"] != "0" else "1"
    second = match_confirmed_profile(
        profile,
        "CKD",
        changed_prohibited_fields,
        calibration,
        "fibrotic-test-release",
    )

    assert first["visual_reference_ids"] == [f"vr_ckd_{index}" for index in range(5)]
    assert second["visual_reference_ids"] == first["visual_reference_ids"]
    assert "p_true" not in str(first)
    assert "tsne" not in str(first).lower()


def test_matching_uses_derived_features_and_balances_features_within_domains():
    profile = confirmed_profile(
        candidate("age", 55, source="age 55"),
        candidate("sex", "female", source="sex female"),
        candidate("height", 180, "cm", "height 180 cm"),
        candidate("weight", 81, "kg", "weight 81 kg"),
        candidate("waist", 80, "cm", "waist 80 cm"),
        candidate("hip", 100, "cm", "hip 100 cm"),
        candidate("blood_pressure", "120/80", source="blood pressure 120/80"),
        candidate("smoking_status", "former", source="smoking former"),
        candidate(
            "alcohol_frequency",
            "once or twice a week",
            source="alcohol once or twice a week",
        ),
        candidate("affected_relative", "yes", source="affected relative yes"),
        candidate("creatinine", 100, "umol/L", "creatinine 100 umol/L"),
        candidate("hba1c", 42, "mmol/mol", "HbA1c 42 mmol/mol"),
    )
    rows = []
    for index in range(5):
        rows.append(
            {
                "visual_reference_id": f"vr_exact_{index}",
                "disease": "MASH",
                "age_recruit": "55",
                "sex": "0",
                "BMI": "25",
                "waist": "80",
                "hip": "100",
                # Raw height and weight are intentionally incompatible. They must
                # not receive extra weight beside the derived BMI match feature.
                "height": "140",
                "weight": "165",
                "DBP": "80",
                "SBP": "120",
                "creatinine": "100",
                "HbA1c": "42",
                "smoking_status": "1",
                "alcohol_freq": "3.0",
                "has_affected_rel": "1",
            }
        )
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "MASH": {
                "eligible_patterns": {
                    "blood_pressure|body_composition|demographics|family_history|lifestyle|optional_laboratory": {
                        "distance_threshold": 0.001
                    }
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "MASH",
        rows,
        calibration,
        "fibrotic-test-release",
    )

    assert result["visual_reference_ids"] == [f"vr_exact_{index}" for index in range(5)]
    assert result["cohort_comparison_result"]["profile_coverage"]["available_domains"] == [
        "demographics",
        "body_composition",
        "blood_pressure",
        "lifestyle",
        "family_history",
        "optional_laboratory",
    ]
    lifestyle = next(
        domain
        for domain in result["aggregate_callout_data"]["domains"]
        if domain["domain"] == "lifestyle"
    )
    assert lifestyle["metrics"] == [
        {
            "feature": "smoking_status",
            "label": "Smoking status",
            "distribution": [{"category": "former", "count": 5}],
        },
        {
            "feature": "alcohol_frequency",
            "label": "Alcohol frequency",
            "distribution": [{"category": "one_to_two_per_week", "count": 5}],
        },
    ]


def test_adaptive_neighborhood_caps_at_twenty_and_keeps_callouts_aggregate_only():
    profile = confirmed_profile(candidate("age", 55, source="age 55"))
    rows = [
        {
            "visual_reference_id": f"vr_{index:02d}",
            "disease": "CKD",
            "age_recruit": "55",
            "sex": "0",
        }
        for index in range(25)
    ]
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "CKD": {
                "eligible_patterns": {
                    "demographics": {"distance_threshold": 0.01}
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "CKD",
        rows,
        calibration,
        "fibrotic-test-release",
    )

    assert list(result) == [
        "dataset_version",
        "cohort_comparison_result",
        "visual_reference_ids",
        "aggregate_callout_data",
    ]
    assert len(result["visual_reference_ids"]) == 20
    assert result["cohort_comparison_result"] == {
        "status": "matched_reference_neighborhood",
        "target": "CKD",
        "profile_coverage": {
            "available_domains": ["demographics"],
            "unavailable_domains": [
                "body_composition",
                "blood_pressure",
                "lifestyle",
                "family_history",
                "optional_laboratory",
            ],
            "eligible": True,
            "calibration_pattern": "demographics",
        },
        "matching_domains": ["demographics"],
        "neighborhood_size": 20,
        "minimum_display_region_size": 5,
        "limitations": [
            "Research cohort comparison only; the Demo Profile is not embedded and no diagnosis, prognosis, or personal outcome is inferred."
        ],
    }
    aggregate = result["aggregate_callout_data"]
    assert aggregate["domains"] == [
        {
            "domain": "demographics",
            "metrics": [
                {
                    "feature": "age",
                    "label": "Age",
                    "unit": "years",
                    "median": 55.0,
                    "range": [55.0, 55.0],
                }
            ],
        }
    ]
    assert "visual_reference" not in str(aggregate).lower()
    assert "risk" not in str(result).lower()
    assert "similarity" not in str(result).lower()


def test_aggregate_callouts_suppress_sparse_categorical_cells():
    profile = confirmed_profile(
        candidate("age", 55, source="age 55"),
        candidate("smoking_status", "former", source="former smoker"),
        candidate("creatinine", 100, "umol/L", "creatinine 100 umol/L"),
    )
    rows = [
        {
            "visual_reference_id": f"vr_{index}",
            "disease": "MASH",
            "age_recruit": "55",
            "smoking_status": "1" if index < 5 else "",
            "creatinine": "100" if index < 5 else "",
        }
        for index in range(6)
    ]
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "MASH": {
                "eligible_patterns": {
                    "demographics|lifestyle|optional_laboratory": {
                        "distance_threshold": 1.0
                    }
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "MASH",
        rows,
        calibration,
        "fibrotic-test-release",
    )

    lifestyle = next(
        domain
        for domain in result["aggregate_callout_data"]["domains"]
        if domain["domain"] == "lifestyle"
    )
    assert lifestyle["metrics"] == [
        {
            "feature": "smoking_status",
            "label": "Smoking status",
            "suppressed": True,
        }
    ]
    optional_laboratory = next(
        domain
        for domain in result["aggregate_callout_data"]["domains"]
        if domain["domain"] == "optional_laboratory"
    )
    assert optional_laboratory["metrics"] == [
        {
            "feature": "creatinine",
            "label": "Creatinine",
            "suppressed": True,
        }
    ]


def test_insufficient_coverage_requests_the_nearest_calibrated_missing_domain():
    profile = confirmed_profile(candidate("age", 55, source="age 55"))
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "MASH": {
                "eligible_patterns": {
                    "demographics|lifestyle": {"distance_threshold": 0.2},
                    "body_composition|demographics|lifestyle": {
                        "distance_threshold": 0.15
                    },
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "MASH",
        [],
        calibration,
        "fibrotic-test-release",
    )

    comparison = result["cohort_comparison_result"]
    assert comparison["status"] == "insufficient_profile_coverage"
    assert comparison["profile_coverage"] == {
        "available_domains": ["demographics"],
        "unavailable_domains": [
            "body_composition",
            "blood_pressure",
            "lifestyle",
            "family_history",
            "optional_laboratory",
        ],
        "eligible": False,
        "coverage_recommendation": {
            "calibration_pattern": "demographics|lifestyle",
            "missing_domains": ["lifestyle"],
            "measurements_by_domain": {
                "lifestyle": ["smoking status", "alcohol frequency"]
            },
        },
    }
    assert result["visual_reference_ids"] == []
    assert result["aggregate_callout_data"] is None


def test_insufficient_coverage_recommends_the_nearest_complete_calibrated_pattern():
    profile = confirmed_profile(candidate("age", 55, source="age 55"))
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "CKD": {
                "eligible_patterns": {
                    "blood_pressure|body_composition|demographics|lifestyle": {
                        "distance_threshold": 0.1,
                        "median_top5_overlap": 0.6,
                        "p10_top5_overlap": 0.2,
                    },
                    "blood_pressure|body_composition|demographics|family_history|lifestyle": {
                        "distance_threshold": 0.1,
                        "median_top5_overlap": 0.8,
                        "p10_top5_overlap": 0.6,
                    },
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "CKD",
        [],
        calibration,
        "fibrotic-test-release",
    )

    recommendation = result["cohort_comparison_result"]["profile_coverage"][
        "coverage_recommendation"
    ]
    assert recommendation["calibration_pattern"] == (
        "blood_pressure|body_composition|demographics|lifestyle"
    )
    assert recommendation["missing_domains"] == [
        "body_composition",
        "blood_pressure",
        "lifestyle",
    ]


def test_eligible_partial_profile_reports_only_calibrated_available_domains():
    profile = confirmed_profile(
        candidate("age", 55, source="age 55"),
        candidate("smoking_status", "former", source="former smoker"),
    )
    rows = [
        {
            "visual_reference_id": f"vr_{index}",
            "disease": "MASH",
            "age_recruit": "55",
            "smoking_status": "1",
            "creatinine": "100",
        }
        for index in range(5)
    ]
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "MASH": {
                "eligible_patterns": {
                    "demographics|lifestyle": {
                        "distance_threshold": 0.1,
                        "median_top5_overlap": 0.8,
                        "p10_top5_overlap": 0.4,
                    }
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "MASH",
        rows,
        calibration,
        "fibrotic-test-release",
    )

    comparison = result["cohort_comparison_result"]
    assert comparison["profile_coverage"] == {
        "available_domains": ["demographics", "lifestyle"],
        "unavailable_domains": [
            "body_composition",
            "blood_pressure",
            "family_history",
            "optional_laboratory",
        ],
        "eligible": True,
        "calibration_pattern": "demographics|lifestyle",
        "masking_stability": {
            "median_top5_overlap": 0.8,
            "p10_top5_overlap": 0.4,
        },
    }
    assert comparison["matching_domains"] == ["demographics", "lifestyle"]
    assert comparison["minimum_display_region_size"] == 5
    assert [
        domain["domain"] for domain in result["aggregate_callout_data"]["domains"]
    ] == ["demographics", "lifestyle"]


def test_threshold_failure_returns_no_stable_neighborhood_without_forced_neighbors():
    profile = confirmed_profile(candidate("age", 33, source="age 33"))
    rows = [
        {
            "visual_reference_id": f"vr_far_{index}",
            "disease": "CKD",
            "age_recruit": "90",
            "sex": "0",
        }
        for index in range(30)
    ]
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "CKD": {
                "eligible_patterns": {
                    "demographics": {"distance_threshold": 0.05}
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "CKD",
        rows,
        calibration,
        "fibrotic-test-release",
    )

    assert result["cohort_comparison_result"]["status"] == "no_stable_neighborhood"
    assert result["cohort_comparison_result"]["neighborhood_size"] == 0
    assert result["visual_reference_ids"] == []
    assert result["aggregate_callout_data"] is None


def test_outside_reference_support_is_preserved_as_an_honest_edge_state():
    profile = confirmed_profile(candidate("age", 32, source="age 32"))
    rows = [
        {
            "visual_reference_id": f"vr_{index}",
            "disease": "CKD",
            "age_recruit": "33",
            "sex": "0",
        }
        for index in range(5)
    ]
    calibration = {
        "methodology": {
            "minimum_group_size": 5,
            "maximum_references": 20,
            "aggregate_cell_suppression_minimum": 5,
        },
        "targets": {
            "CKD": {
                "eligible_patterns": {
                    "demographics": {"distance_threshold": 0.001}
                }
            }
        },
    }

    result = match_confirmed_profile(
        profile,
        "CKD",
        rows,
        calibration,
        "fibrotic-test-release",
    )

    comparison = result["cohort_comparison_result"]
    assert comparison["status"] == "no_stable_neighborhood"
    assert comparison["profile_coverage"]["outside_reference_support_domains"] == [
        "demographics"
    ]
    assert result["visual_reference_ids"] == []


def test_calibration_records_masking_stability_and_fifth_neighbor_thresholds():
    rows = []
    for index in range(12):
        rows.append(
            {
                "visual_reference_id": f"vr_{index}",
                "disease": "CKD",
                "age_recruit": str(40 + index),
                "sex": str(index % 2),
                "BMI": str(20 + index),
                "waist": str(80 + index),
                "hip": str(100 + index),
                "height": str(160 + index),
                "weight": str(60 + index),
                "DBP": str(70 + index),
                "SBP": str(110 + index),
                "creatinine": str(60 + index),
                "HbA1c": str(30 + index),
                "smoking_status": str(index % 3),
                "alcohol_freq": str((index % 6) + 1),
                "has_affected_rel": str(index % 2),
            }
        )

    calibration = calibrate_matching(rows, "fibrotic-test-release")

    assert calibration["dataset_version"] == "fibrotic-test-release"
    assert calibration["methodology"] == {
        "reference_neighbor_count": 5,
        "minimum_group_size": 5,
        "maximum_references": 20,
        "aggregate_cell_suppression_minimum": 5,
        "stability_median_overlap_minimum": 0.6,
        "stability_p10_overlap_minimum": 0.2,
        "distance_threshold_quantile": 0.95,
    }
    target = calibration["targets"]["CKD"]
    assert target["reference_count"] == 12
    pattern = target["eligible_patterns"][
        "blood_pressure|body_composition|demographics|optional_laboratory"
    ]
    assert pattern["median_top5_overlap"] >= 0.6
    assert pattern["p10_top5_overlap"] >= 0.2
    assert 0 < pattern["distance_threshold"] <= 1


@pytest.mark.asyncio
async def test_match_endpoint_revalidates_confirmed_candidates_and_requires_a_supported_target(
    api_client,
):
    profile = confirmed_profile(candidate("age", 55, source="age 55"))
    profile["reported_features"]["age"]["normalized_value"] = 90
    rows = [
        {
            "visual_reference_id": f"vr_age_55_{index}",
            "disease": "CKD",
            "age_recruit": "55",
            "sex": "0",
        }
        for index in range(5)
    ]
    release = {
        "dataset_version": "fibrotic-test-release",
        "rows": rows,
        "calibration": {
            "methodology": {
                "minimum_group_size": 5,
                "maximum_references": 20,
                "aggregate_cell_suppression_minimum": 5,
            },
            "targets": {
                "CKD": {
                    "eligible_patterns": {
                        "demographics": {"distance_threshold": 0.01}
                    }
                }
            },
        },
    }
    app.dependency_overrides[get_profile_matching_release] = lambda: release

    response = await api_client.post(
        "/profile/match",
        json={"confirmed_profile": profile, "target": "CKD"},
    )
    unsupported = await api_client.post(
        "/profile/match",
        json={"confirmed_profile": profile, "target": "Diabetes"},
    )

    assert response.status_code == 200
    assert response.json()["visual_reference_ids"] == [
        f"vr_age_55_{index}" for index in range(5)
    ]
    assert unsupported.status_code == 422


def test_matching_release_requires_exact_private_schema_hash_and_visual_id_mapping(
    tmp_path,
):
    private_path = tmp_path / "private.csv"
    display_path = tmp_path / "display.csv"
    manifest_path = tmp_path / "manifest.json"
    calibration_path = tmp_path / "calibration.json"
    private_row = {field: "1" for field in MATCH_FIELDS}
    private_row.update(visual_reference_id="vr_one", disease="CKD")
    public_row = {field: "1" for field in PUBLIC_FIELDS}
    public_row.update(visual_reference_id="vr_one", disease="CKD")

    def write_csv(path, fieldnames, row):
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    write_csv(private_path, MATCH_FIELDS, private_row)
    write_csv(display_path, PUBLIC_FIELDS, public_row)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "fibrotic-test-release",
                "private_schema": MATCH_FIELDS,
                "private_sha256": hashlib.sha256(private_path.read_bytes()).hexdigest(),
            }
        )
    )
    calibration_path.write_text(
        json.dumps(
            {
                "dataset_version": "fibrotic-test-release",
                "methodology": {},
                "targets": {},
            }
        )
    )

    release = read_matching_release(
        private_path,
        display_path,
        manifest_path,
        calibration_path,
    )
    assert release["rows"][0]["visual_reference_id"] == "vr_one"

    public_row["visual_reference_id"] = "vr_other"
    write_csv(display_path, PUBLIC_FIELDS, public_row)
    with pytest.raises(RuntimeError, match="one-to-one"):
        read_matching_release(
            private_path,
            display_path,
            manifest_path,
            calibration_path,
        )

"""Deterministic profile matching for the fibrotic reference cohort."""

import csv
import hashlib
import json
import os
from collections import Counter
from functools import lru_cache
from pathlib import Path
from statistics import median

from fibrotic_contract import MATCH_FIELDS, PUBLIC_FIELDS, TARGETS


CONTINUOUS_RANGES = {
    "age": (33.0, 90.0),
    "bmi": (15.0, 60.0),
    "waist_to_hip_ratio": (0.6218487394957983, 1.3058823529411765),
    "sbp": (80.0, 220.0),
    "dbp": (46.0, 120.0),
    "creatinine": (20.4, 500.0),
    "hba1c": (20.0, 113.0),
}
CATEGORY_VALUES = {
    "sex": {"female": "0", "male": "1"},
    "smoking_status": {"never": "0", "former": "1", "current": "2"},
    "alcohol_frequency": {
        "daily_or_almost_daily": "1",
        "three_to_four_per_week": "2",
        "one_to_two_per_week": "3",
        "one_to_three_per_month": "4",
        "special_occasions": "5",
        "never": "6",
    },
    "affected_relative": {False: "0", True: "1"},
}
DOMAIN_FEATURES = {
    "demographics": (
        ("age", "age_recruit", "continuous"),
        ("sex", "sex", "categorical"),
    ),
    "body_composition": (
        ("bmi", "BMI", "continuous"),
        ("waist_to_hip_ratio", "waist_to_hip_ratio", "continuous"),
    ),
    "blood_pressure": (
        ("sbp", "SBP", "continuous"),
        ("dbp", "DBP", "continuous"),
    ),
    "lifestyle": (
        ("smoking_status", "smoking_status", "categorical"),
        ("alcohol_frequency", "alcohol_freq", "categorical"),
    ),
    "family_history": (
        ("affected_relative", "has_affected_rel", "categorical"),
    ),
    "optional_laboratory": (
        ("creatinine", "creatinine", "continuous"),
        ("hba1c", "HbA1c", "continuous"),
    ),
}
DOMAIN_MEASUREMENTS = {
    "demographics": ["age", "sex when supplied"],
    "body_composition": ["height and weight", "BMI", "waist and hip circumference"],
    "blood_pressure": ["systolic and diastolic blood pressure"],
    "lifestyle": ["smoking status", "alcohol frequency"],
    "family_history": ["affected-relative status"],
    "optional_laboratory": ["creatinine", "HbA1c"],
}
FEATURE_PRESENTATION = {
    "age": ("Age", "years"),
    "bmi": ("BMI", "kg/m²"),
    "waist_to_hip_ratio": ("Waist-to-hip ratio", "ratio"),
    "sbp": ("Systolic blood pressure", "mmHg"),
    "dbp": ("Diastolic blood pressure", "mmHg"),
    "creatinine": ("Creatinine", "µmol/L"),
    "hba1c": ("HbA1c", "mmol/mol"),
    "sex": ("Sex", None),
    "smoking_status": ("Smoking status", None),
    "alcohol_frequency": ("Alcohol frequency", None),
    "affected_relative": ("Affected relative", None),
}


def resolve_private_matching_path(data_dir, download_file=None):
    dataset_repo = os.environ.get("FIBROTIC_MATCH_DATASET_REPO")
    if not dataset_repo:
        return Path(data_dir) / "private" / "fibrotic_match.csv"

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN secret is required for the configured private matching dataset"
        )
    if download_file is None:
        from huggingface_hub import hf_hub_download

        download_file = hf_hub_download
    try:
        path = download_file(
            repo_id=dataset_repo,
            filename="fibrotic_match.csv",
            repo_type="dataset",
            revision=os.environ.get("FIBROTIC_MATCH_DATASET_REVISION", "main"),
            token=token,
        )
    except Exception as error:
        raise RuntimeError(
            "Private matching dataset could not be downloaded"
        ) from error
    return Path(path)


@lru_cache(maxsize=1)
def load_matching_release():
    data_dir = Path(__file__).parent / "data"
    return read_matching_release(
        resolve_private_matching_path(data_dir),
        data_dir / "fibrotic_release" / "fibrotic_embedding.csv",
        data_dir / "fibrotic_release" / "fibrotic_manifest.json",
        data_dir / "fibrotic_matching_calibration.json",
    )


def read_matching_release(private_path, display_path, manifest_path, calibration_path):
    manifest = json.loads(Path(manifest_path).read_text())
    calibration = json.loads(Path(calibration_path).read_text())
    private_path = Path(private_path)
    with private_path.open(newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        private_schema = reader.fieldnames
    with Path(display_path).open(newline="") as file:
        reader = csv.DictReader(file)
        display_rows = list(reader)
        display_schema = reader.fieldnames

    if private_schema != MATCH_FIELDS or manifest.get("private_schema") != MATCH_FIELDS:
        raise RuntimeError("Fibrotic private matching schema is not approved")
    if display_schema != PUBLIC_FIELDS:
        raise RuntimeError("Fibrotic display schema is not approved")
    private_digest = hashlib.sha256(private_path.read_bytes()).hexdigest()
    if private_digest != manifest.get("private_sha256"):
        raise RuntimeError("Fibrotic private artifact does not match its manifest")
    if calibration["dataset_version"] != manifest["dataset_version"]:
        raise RuntimeError("Matching calibration belongs to another Dataset Release")
    private_ids = [row["visual_reference_id"] for row in rows]
    display_ids = [row["visual_reference_id"] for row in display_rows]
    if (
        len(private_ids) != len(set(private_ids))
        or len(display_ids) != len(set(display_ids))
        or set(private_ids) != set(display_ids)
    ):
        raise RuntimeError(
            "Private and display Visual Reference IDs must have a one-to-one mapping"
        )
    if not {row["disease"] for row in rows} <= TARGETS:
        raise RuntimeError("Private artifact contains an unsupported comparison target")
    return {
        "dataset_version": manifest["dataset_version"],
        "rows": rows,
        "calibration": calibration,
    }


def _profile_values(confirmed_profile):
    values = {}
    for field, feature in confirmed_profile.get("reported_features", {}).items():
        if feature.get("status") not in {"valid", "outside_reference_support"}:
            continue
        value = feature.get("normalized_value")
        if field in CATEGORY_VALUES:
            value = CATEGORY_VALUES[field].get(value)
        if value is not None:
            values[field] = value
    derived = confirmed_profile.get("derived_features", {})
    for field in ("bmi", "waist_to_hip_ratio"):
        feature = derived.get(field)
        if feature and feature.get("status") in {"valid", "outside_reference_support"}:
            values[field] = feature.get("value")
    return values


def _outside_reference_support_domains(confirmed_profile):
    domains = set()
    for section in ("reported_features", "derived_features"):
        for feature in confirmed_profile.get(section, {}).values():
            if feature.get("status") == "outside_reference_support" and feature.get(
                "domain"
            ) in DOMAIN_FEATURES:
                domains.add(feature["domain"])
    return [domain for domain in DOMAIN_FEATURES if domain in domains]


def _row_value(row, row_field):
    if row_field == "waist_to_hip_ratio":
        if not row.get("waist") or not row.get("hip") or float(row["hip"]) == 0:
            return None
        return round(float(row["waist"]) / float(row["hip"]), 2)
    value = row.get(row_field)
    if value in {None, "", "-3"}:
        return None
    return value


def _category_code(value):
    text = str(value)
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else str(number)


def _distance(profile_values, row):
    domain_distances = []
    for features in DOMAIN_FEATURES.values():
        differences = []
        for profile_field, row_field, kind in features:
            row_value = _row_value(row, row_field)
            if profile_field not in profile_values or row_value is None:
                continue
            if kind == "continuous":
                low, high = CONTINUOUS_RANGES[profile_field]
                difference = abs(float(profile_values[profile_field]) - float(row_value)) / (high - low)
            else:
                difference = (
                    0.0
                    if _category_code(profile_values[profile_field])
                    == _category_code(row_value)
                    else 1.0
                )
            differences.append(difference)
        if differences:
            domain_distances.append(sum(differences) / len(differences))
    if not domain_distances:
        return None
    return sum(domain_distances) / len(domain_distances)


def _aggregate_callout(profile_values, selected_rows, suppression_minimum):
    domains = []
    for domain, features in DOMAIN_FEATURES.items():
        metrics = []
        for profile_field, row_field, kind in features:
            if profile_field not in profile_values:
                continue
            if kind == "categorical":
                reverse = {
                    _category_code(encoded): category
                    for category, encoded in CATEGORY_VALUES[profile_field].items()
                }
                counts = Counter(
                    reverse.get(_category_code(value), str(value))
                    for row in selected_rows
                    if (value := _row_value(row, row_field)) is not None
                )
                if counts:
                    label, _ = FEATURE_PRESENTATION[profile_field]
                    metric = {"feature": profile_field, "label": label}
                    missing_count = len(selected_rows) - sum(counts.values())
                    has_sparse_missing_cell = 0 < missing_count < suppression_minimum
                    if (
                        min(counts.values()) < suppression_minimum
                        or has_sparse_missing_cell
                    ):
                        metric["suppressed"] = True
                    else:
                        metric["distribution"] = [
                            {"category": category, "count": count}
                            for category, count in sorted(
                                counts.items(), key=lambda item: str(item[0])
                            )
                        ]
                    metrics.append(metric)
                continue
            values = [
                float(value)
                for row in selected_rows
                if (value := _row_value(row, row_field)) is not None
            ]
            if not values:
                continue
            label, unit = FEATURE_PRESENTATION[profile_field]
            missing_count = len(selected_rows) - len(values)
            has_sparse_missing_cell = 0 < missing_count < suppression_minimum
            if len(values) < suppression_minimum or has_sparse_missing_cell:
                metrics.append(
                    {
                        "feature": profile_field,
                        "label": label,
                        "suppressed": True,
                    }
                )
                continue
            metrics.append(
                {
                    "feature": profile_field,
                    "label": label,
                    "unit": unit,
                    "median": round(float(median(values)), 2),
                    "range": [round(min(values), 2), round(max(values), 2)],
                }
            )
        if metrics:
            domains.append({"domain": domain, "metrics": metrics})
    return {
        "reference_count": len(selected_rows),
        "title": "Matched reference neighborhood",
        "description": "Release-controlled aggregate context; sparse cells are suppressed.",
        "domains": domains,
    }


def _coverage_recommendation(available_domains, eligible_patterns):
    available = set(available_domains)
    candidates = []
    for pattern, evidence in eligible_patterns.items():
        domains = set(pattern.split("|"))
        if not available <= domains:
            continue
        missing = domains - available
        if not missing:
            continue
        candidates.append(
            (
                len(missing),
                -evidence.get("p10_top5_overlap", 0),
                -evidence.get("median_top5_overlap", 0),
                pattern,
                missing,
            )
        )
    if not candidates:
        return None
    _, _, _, pattern, missing = min(candidates)
    ordered_missing = [domain for domain in DOMAIN_FEATURES if domain in missing]
    return {
        "calibration_pattern": pattern,
        "missing_domains": ordered_missing,
        "measurements_by_domain": {
            domain: DOMAIN_MEASUREMENTS[domain] for domain in ordered_missing
        },
    }


def evaluate_profile_coverage(profile, target, calibration):
    """Target-aware Profile Coverage for a draft or confirmed profile.

    Domain-based and deterministic. It reports which feature domains are
    represented, whether the profile satisfies a reviewed eligible pattern
    for the target, and — when it does not — the nearest optional additions
    that would reach one. It never returns a completion percentage, match
    accuracy, or confidence, and no single field is universally required.
    Both the pre-confirmation coverage endpoint and match_confirmed_profile
    call this, so review-time guidance and matching agree by construction.
    """
    values = _profile_values(profile)
    available_domains = [
        domain
        for domain, features in DOMAIN_FEATURES.items()
        if any(feature[0] in values for feature in features)
    ]
    pattern = "|".join(sorted(available_domains))
    unavailable_domains = [
        domain for domain in DOMAIN_FEATURES if domain not in available_domains
    ]
    eligible_patterns = calibration.get("targets", {}).get(target, {}).get(
        "eligible_patterns", {}
    )
    pattern_calibration = eligible_patterns.get(pattern)
    coverage = {
        "available_domains": available_domains,
        "unavailable_domains": unavailable_domains,
        "eligible": bool(pattern_calibration),
    }
    if pattern_calibration:
        coverage["calibration_pattern"] = pattern
    else:
        recommendation = _coverage_recommendation(available_domains, eligible_patterns)
        if recommendation:
            coverage["coverage_recommendation"] = recommendation
    outside_support_domains = _outside_reference_support_domains(profile)
    if outside_support_domains:
        coverage["outside_reference_support_domains"] = outside_support_domains
    return coverage, pattern_calibration


def match_confirmed_profile(confirmed_profile, target, rows, calibration, dataset_version):
    values = _profile_values(confirmed_profile)
    coverage, pattern_calibration = evaluate_profile_coverage(
        confirmed_profile, target, calibration
    )
    available_domains = coverage["available_domains"]
    if not pattern_calibration:
        return {
            "dataset_version": dataset_version,
            "cohort_comparison_result": {
                "status": "insufficient_profile_coverage",
                "target": target,
                "profile_coverage": coverage,
            },
            "visual_reference_ids": [],
            "aggregate_callout_data": None,
        }

    qualified = []
    threshold = pattern_calibration["distance_threshold"]
    for row in rows:
        if row.get("disease") != target:
            continue
        distance = _distance(values, row)
        if distance is not None and distance <= threshold:
            qualified.append((distance, row["visual_reference_id"], row))
    qualified.sort(key=lambda item: (item[0], item[1]))
    maximum = calibration["methodology"]["maximum_references"]
    selected = qualified[:maximum]
    visual_reference_ids = [reference_id for _, reference_id, _ in selected]
    minimum = calibration["methodology"]["minimum_group_size"]
    status = (
        "matched_reference_neighborhood"
        if len(visual_reference_ids) >= minimum
        else "no_stable_neighborhood"
    )
    if status == "no_stable_neighborhood":
        visual_reference_ids = []
    profile_coverage = coverage
    masking_stability = {
        key: pattern_calibration[key]
        for key in ("median_top5_overlap", "p10_top5_overlap")
        if key in pattern_calibration
    }
    if masking_stability:
        profile_coverage["masking_stability"] = masking_stability
    return {
        "dataset_version": dataset_version,
        "cohort_comparison_result": {
            "status": status,
            "target": target,
            "profile_coverage": profile_coverage,
            "matching_domains": available_domains,
            "neighborhood_size": len(visual_reference_ids),
            "minimum_display_region_size": calibration["methodology"][
                "aggregate_cell_suppression_minimum"
            ],
            "limitations": [
                "Research cohort comparison only; the Demo Profile is not embedded and no diagnosis, prognosis, or personal outcome is inferred."
            ],
        },
        "visual_reference_ids": visual_reference_ids,
        "aggregate_callout_data": (
            _aggregate_callout(
                values,
                [row for _, _, row in selected],
                calibration["methodology"]["aggregate_cell_suppression_minimum"],
            )
            if visual_reference_ids
            else None
        ),
    }

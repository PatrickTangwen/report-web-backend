"""Generate evidence-driven matching coverage and threshold parameters."""

import argparse
import csv
import itertools
import json
import warnings
from pathlib import Path

import numpy as np

from profile_matching import CONTINUOUS_RANGES, DOMAIN_FEATURES, _row_value


METHODOLOGY = {
    "reference_neighbor_count": 5,
    "minimum_group_size": 5,
    "maximum_references": 20,
    "aggregate_cell_suppression_minimum": 5,
    "stability_median_overlap_minimum": 0.6,
    "stability_p10_overlap_minimum": 0.2,
    "distance_threshold_quantile": 0.95,
}


def _feature_matrix(rows, profile_field, row_field, kind):
    values = [_row_value(row, row_field) for row in rows]
    if kind == "continuous":
        array = np.array(
            [float(value) if value is not None else np.nan for value in values],
            dtype=float,
        )
        low, high = CONTINUOUS_RANGES[profile_field]
        return np.abs(array[:, None] - array[None, :]) / (high - low)
    array = np.array(values, dtype=object)
    available = np.array([value is not None for value in values])
    differences = (array[:, None] != array[None, :]).astype(float)
    differences[~(available[:, None] & available[None, :])] = np.nan
    return differences


def _domain_matrices(rows):
    matrices = {}
    for domain, features in DOMAIN_FEATURES.items():
        feature_matrices = [
            _feature_matrix(rows, profile_field, row_field, kind)
            for profile_field, row_field, kind in features
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            matrices[domain] = np.nanmean(feature_matrices, axis=0)
    return matrices


def _combined_distance(matrices, domains):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        distance = np.nanmean([matrices[domain] for domain in domains], axis=0)
    np.fill_diagonal(distance, np.inf)
    return distance


def _neighbor_indices(distance, count):
    return np.argpartition(distance, count - 1, axis=1)[:, :count]


def _pattern_evidence(distance, full_neighbors, neighbor_count):
    neighbors = _neighbor_indices(distance, neighbor_count)
    overlaps = np.array(
        [
            len(set(masked) & set(full)) / neighbor_count
            for masked, full in zip(neighbors, full_neighbors)
        ]
    )
    fifth_distances = np.partition(distance, neighbor_count - 1, axis=1)[
        :, neighbor_count - 1
    ]
    finite = fifth_distances[np.isfinite(fifth_distances)]
    if not len(finite):
        return None
    return {
        "median_top5_overlap": round(float(np.median(overlaps)), 4),
        "p10_top5_overlap": round(float(np.quantile(overlaps, 0.1)), 4),
        "distance_threshold": round(
            float(
                np.quantile(
                    finite,
                    METHODOLOGY["distance_threshold_quantile"],
                )
            ),
            6,
        ),
    }


def calibrate_matching(rows, dataset_version):
    targets = {}
    neighbor_count = METHODOLOGY["reference_neighbor_count"]
    for target in sorted({row["disease"] for row in rows}):
        target_rows = [row for row in rows if row["disease"] == target]
        if len(target_rows) <= neighbor_count:
            raise ValueError(
                f"{target} needs more than {neighbor_count} rows for calibration"
            )
        matrices = _domain_matrices(target_rows)
        domains = list(DOMAIN_FEATURES)
        full_distance = _combined_distance(matrices, domains)
        full_neighbors = _neighbor_indices(full_distance, neighbor_count)
        experiments = {}
        eligible = {}
        for count in range(1, len(domains) + 1):
            for combination in itertools.combinations(domains, count):
                pattern = "|".join(sorted(combination))
                distance = _combined_distance(matrices, combination)
                evidence = _pattern_evidence(
                    distance,
                    full_neighbors,
                    neighbor_count,
                )
                if evidence is None:
                    continue
                experiments[pattern] = evidence
                if (
                    evidence["median_top5_overlap"]
                    >= METHODOLOGY["stability_median_overlap_minimum"]
                    and evidence["p10_top5_overlap"]
                    >= METHODOLOGY["stability_p10_overlap_minimum"]
                ):
                    eligible[pattern] = evidence
        targets[target] = {
            "reference_count": len(target_rows),
            "eligible_patterns": eligible,
            "masking_experiments": experiments,
        }
    return {
        "dataset_version": dataset_version,
        "methodology": METHODOLOGY,
        "continuous_ranges": {
            field: list(bounds) for field, bounds in CONTINUOUS_RANGES.items()
        },
        "targets": targets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--private-artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.private_artifact.open(newline="") as file:
        rows = list(csv.DictReader(file))
    manifest = json.loads(args.manifest.read_text())
    calibration = calibrate_matching(rows, manifest["dataset_version"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(calibration, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Reviewed Synthetic Example Profiles for target-first Demo Profile entry.

Each Comparison Target has exactly one fixed, hand-authored example: a
reviewed list of Feature Candidates (not LLM output, not copied from a
Reference Patient) that the frontend loads and runs through the same
build_profile_draft/confirm_profile pipeline as any visitor-entered
profile. See docs/adr/0008-use-a-guided-form-for-demo-profiles.md.
"""

import json
from functools import lru_cache
from pathlib import Path

from fibrotic_contract import TARGETS

DATA_PATH = Path(__file__).parent / "data" / "synthetic_example_profiles.json"


@lru_cache(maxsize=1)
def load_synthetic_example_profiles():
    payload = json.loads(DATA_PATH.read_text())
    profiles = payload.get("profiles", {})

    missing = TARGETS - set(profiles)
    if missing:
        raise RuntimeError(f"Synthetic Example Profiles missing targets: {sorted(missing)}")
    extra = set(profiles) - TARGETS
    if extra:
        raise RuntimeError(f"Synthetic Example Profiles reference unknown targets: {sorted(extra)}")
    for target, profile in profiles.items():
        if profile.get("target") != target:
            raise RuntimeError(f"Synthetic Example Profile target mismatch for {target}")
        if not profile.get("candidates"):
            raise RuntimeError(f"Synthetic Example Profile for {target} has no candidates")

    return {
        "version": payload["version"],
        "review_status": payload["review_status"],
        "profiles": profiles,
    }


def get_synthetic_example_profile(target):
    if target not in TARGETS:
        raise KeyError(target)
    release = load_synthetic_example_profiles()
    profile = release["profiles"][target]
    return {
        "target": target,
        "version": release["version"],
        "review_status": release["review_status"],
        "candidates": profile["candidates"],
    }

import json
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import app
from demo_profile import build_profile_draft, confirm_profile
from fibrotic_contract import TARGETS
from profile_matching import DOMAIN_FEATURES, _profile_values
from synthetic_example_profiles import (
    get_synthetic_example_profile,
    load_synthetic_example_profiles,
)

CALIBRATION = json.loads(
    (Path(__file__).parent / "data" / "fibrotic_matching_calibration.json").read_text()
)


@pytest_asyncio.fixture
async def api_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def test_exactly_the_seven_approved_targets_have_a_reviewed_example():
    release = load_synthetic_example_profiles()
    assert set(release["profiles"].keys()) == TARGETS
    assert release["review_status"] == "researcher-reviewed"


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_each_synthetic_example_profile_passes_the_real_validation_and_confirmation_pipeline(target):
    """Examples must satisfy the same public contract as a visitor-entered
    profile, not a fixture that bypasses it (spec Testing Decisions)."""
    example = get_synthetic_example_profile(target)
    assert example["target"] == target

    draft = build_profile_draft(example["candidates"])
    assert draft["state"] == "draft"
    assert draft["can_confirm"] is True
    blocking = {"ambiguous", "out_of_range", "conflicting"}
    reported_statuses = {f["status"] for f in draft["reported_features"].values()}
    assert not (reported_statuses & blocking)

    confirmed = confirm_profile(draft)
    assert confirmed["state"] == "confirmed"
    assert confirmed["matching_started"] is False


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_each_synthetic_example_profile_demonstrates_an_eligible_target_specific_coverage_pattern(target):
    """PRD user story #30: each example must demonstrate an eligible
    calibrated coverage pattern for its own target, not just be
    confirmable — otherwise loading it always dead-ends in Coverage
    needed rather than a real matched/no-stable-neighborhood outcome."""
    example = get_synthetic_example_profile(target)
    confirmed = confirm_profile(build_profile_draft(example["candidates"]))
    values = _profile_values(confirmed)
    available_domains = [
        domain
        for domain, features in DOMAIN_FEATURES.items()
        if any(feature[0] in values for feature in features)
    ]
    pattern = "|".join(sorted(available_domains))
    eligible_patterns = CALIBRATION["targets"][target]["eligible_patterns"]
    assert pattern in eligible_patterns, (
        f"{target} Synthetic Example Profile pattern '{pattern}' is not an "
        f"eligible calibrated pattern: {sorted(eligible_patterns)}"
    )


def test_synthetic_example_profile_is_not_generated_by_an_llm_or_copied_from_a_reference_patient():
    release = load_synthetic_example_profiles()
    assert "llm" not in release["version"].lower()
    for target, profile in release["profiles"].items():
        for candidate in profile["candidates"]:
            assert set(candidate) == {
                "field",
                "raw_value",
                "raw_unit",
                "source_text",
                "operation",
            }


def test_unknown_target_is_rejected():
    with pytest.raises(KeyError):
        get_synthetic_example_profile("not_a_real_target")


@pytest.mark.asyncio
@pytest.mark.parametrize("target", sorted(TARGETS))
async def test_api_returns_the_reviewed_example_for_each_target(api_client, target):
    resp = await api_client.get(f"/profile/synthetic-example/{target}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["target"] == target
    assert body["review_status"] == "researcher-reviewed"
    assert body["version"]
    assert len(body["candidates"]) > 0


@pytest.mark.asyncio
async def test_api_rejects_an_unknown_target(api_client):
    resp = await api_client.get("/profile/synthetic-example/not_a_real_target")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_loading_an_example_only_returns_candidates_not_a_confirmed_or_matched_state(api_client):
    """Loading an example must never itself confirm the draft or start
    comparison — the response is raw Feature Candidates only."""
    resp = await api_client.get("/profile/synthetic-example/MASH")
    body = resp.json()
    assert set(body.keys()) == {"target", "version", "review_status", "candidates"}
    assert "state" not in body
    assert "matching_started" not in body

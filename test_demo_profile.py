import json
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app import app
from demo_profile import (
    INPUT_BOUNDS,
    REFERENCE_SUPPORT,
    ProfileRateLimiter,
    build_profile_draft,
    confirm_profile,
    parse_feature_candidates,
)


def candidate(field, value, unit, source_text, operation="set"):
    return {
        "field": field,
        "raw_value": value,
        "raw_unit": unit,
        "source_text": source_text,
        "operation": operation,
    }


@pytest_asyncio.fixture
async def api_client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def test_height_and_weight_create_visible_bmi_before_explicit_confirmation():
    draft = build_profile_draft(
        [
            candidate("height", 5.5, "ft", "height is 5.5 feet"),
            candidate("weight", 150, "lb", "weight is 150 pounds"),
        ]
    )

    assert draft["state"] == "draft"
    assert draft["can_confirm"] is True
    assert draft["reported_features"]["height"]["normalized_value"] == 167.64
    assert draft["reported_features"]["height"]["source_text"] == "height is 5.5 feet"
    assert draft["reported_features"]["weight"]["original_unit"] == "lb"
    assert draft["derived_features"]["bmi"] == {
        "label": "BMI",
        "value": 24.2,
        "unit": "kg/m²",
        "domain": "body_composition",
        "status": "valid",
        "derived_from": ["height", "weight"],
    }

    confirmed = confirm_profile(draft)

    assert confirmed["state"] == "confirmed"
    assert confirmed["matching_started"] is False
    assert confirmed["derived_features"]["bmi"]["value"] == 24.2


def test_units_and_reference_support_are_deterministic_without_clamping():
    draft = build_profile_draft(
        [
            candidate("height", 70, "in", "70 inches"),
            candidate("weight", 400, "lb", "400 pounds"),
            candidate("creatinine", 1.2, "mg/dL", "creatinine 1.2 mg/dL"),
            candidate("hba1c", 6.5, "%", "HbA1c is 6.5%"),
        ]
    )

    assert draft["reported_features"]["height"]["normalized_value"] == 177.8
    assert draft["reported_features"]["weight"]["normalized_value"] == 181.44
    assert draft["reported_features"]["weight"]["status"] == "outside_reference_support"
    assert draft["reported_features"]["creatinine"]["normalized_value"] == 106.08
    assert draft["reported_features"]["hba1c"]["normalized_value"] == 47.5
    assert draft["can_confirm"] is True


def test_compound_feet_and_inches_are_parsed_without_llm_arithmetic():
    draft = build_profile_draft(
        [candidate("height", "5 ft 6 in", "ft/in", "height is 5 ft 6 in")]
    )

    assert draft["reported_features"]["height"]["normalized_value"] == 167.64
    assert draft["reported_features"]["height"]["original_value"] == "5 ft 6 in"


def test_ambiguous_invalid_and_unsupported_candidates_remain_visible():
    draft = build_profile_draft(
        [
            candidate("weight", 80, None, "weight 80"),
            candidate("age", -4, None, "age negative four"),
            candidate("favorite_color", "blue", None, "favorite color blue"),
        ]
    )

    assert draft["reported_features"]["weight"]["status"] == "ambiguous"
    assert draft["reported_features"]["age"]["status"] == "out_of_range"
    assert draft["reported_features"]["favorite_color"]["status"] == "unsupported"
    assert draft["can_confirm"] is False
    assert "height" in draft["missing_fields"]
    assert "bmi" not in draft["derived_features"]


def test_llm_unit_must_be_supported_by_the_candidate_source_text():
    invented = build_profile_draft(
        [candidate("weight", 80, "kg", "weight 80")]
    )
    explicit = build_profile_draft(
        [candidate("weight", 180, "lb", "weight is 180 pounds")]
    )

    assert invented["reported_features"]["weight"]["status"] == "ambiguous"
    assert "not explicit" in invented["reported_features"]["weight"]["message"]
    assert explicit["reported_features"]["weight"]["status"] == "valid"


def test_schema_assigns_only_unambiguous_units_when_llm_leaves_them_empty():
    draft = build_profile_draft(
        [
            candidate("age", 55, None, "I am 55"),
            candidate("bmi", 24.5, None, "BMI 24.5"),
            candidate("blood_pressure", "135/85", None, "blood pressure 135/85"),
        ]
    )

    assert draft["reported_features"]["age"]["normalized_unit"] == "years"
    assert draft["reported_features"]["bmi"]["normalized_unit"] == "kg/m²"
    assert draft["reported_features"]["sbp"]["normalized_unit"] == "mmHg"


def test_llm_cannot_invent_an_implicit_field_unit():
    draft = build_profile_draft(
        [candidate("bmi", 30, "kg/m2", "BMI 30")]
    )

    assert draft["reported_features"]["bmi"]["status"] == "ambiguous"


def test_blood_pressure_and_waist_hip_components_are_visible_before_derivation():
    draft = build_profile_draft(
        [
            candidate("blood_pressure", "135/85", None, "blood pressure 135/85"),
            candidate("waist", 32, "in", "waist 32 inches"),
            candidate("hip", 40, "in", "hip 40 inches"),
        ]
    )

    assert draft["reported_features"]["sbp"]["normalized_value"] == 135
    assert draft["reported_features"]["dbp"]["normalized_value"] == 85
    assert draft["reported_features"]["sbp"]["source_text"] == "blood pressure 135/85"
    assert draft["derived_features"]["waist_to_hip_ratio"]["value"] == 0.8


def test_conflict_requires_an_explicit_correction_and_keeps_candidate_history():
    first = candidate("age", 55, None, "I am 55")
    conflicting = candidate("age", 56, None, "age 56")
    draft = build_profile_draft([first, conflicting])

    assert draft["reported_features"]["age"]["status"] == "conflicting"
    assert draft["reported_features"]["age"]["alternatives"] == [55, 56]
    assert draft["can_confirm"] is False

    corrected = build_profile_draft(
        [first, conflicting, candidate("age", 57, None, "Correction: 57", "correct")]
    )

    assert corrected["reported_features"]["age"]["status"] == "valid"
    assert corrected["reported_features"]["age"]["normalized_value"] == 57
    assert [item["source_text"] for item in corrected["candidates"]] == [
        "I am 55",
        "age 56",
        "Correction: 57",
    ]
    assert corrected["reported_features"]["age"]["source_history"][0]["source_text"] == "I am 55"
    assert corrected["reported_features"]["age"]["source_history"][-1]["source_text"] == "Correction: 57"


def test_reported_bmi_must_agree_with_the_calculated_value():
    draft = build_profile_draft(
        [
            candidate("height", 180, "cm", "height 180 cm"),
            candidate("weight", 81, "kg", "weight 81 kg"),
            candidate("bmi", 30, None, "BMI 30"),
        ]
    )

    assert draft["derived_features"]["bmi"]["value"] == 25.0
    assert draft["reported_features"]["bmi"]["status"] == "conflicting"
    assert draft["reported_features"]["bmi"]["calculated_value"] == 25.0
    assert draft["can_confirm"] is False


def test_reported_bmi_mismatch_can_be_explicitly_confirmed():
    candidates = [
        candidate("height", 180, "cm", "height 180 cm"),
        candidate("weight", 81, "kg", "weight 81 kg"),
        candidate("bmi", 30, "kg/m2", "BMI 30 kg/m2"),
        candidate("bmi", 30, None, "Edited in review: BMI", "correct"),
    ]

    draft = build_profile_draft(candidates)

    assert draft["reported_features"]["bmi"]["status"] == "valid"
    assert draft["reported_features"]["bmi"]["calculated_value"] == 25.0
    assert draft["reported_features"]["bmi"]["mismatch_acknowledged"] is True
    assert draft["can_confirm"] is True


def test_implausible_derived_bmi_is_visible_and_blocks_confirmation():
    draft = build_profile_draft(
        [
            candidate("height", 50, "cm", "height 50 cm"),
            candidate("weight", 500, "kg", "weight 500 kg"),
        ]
    )

    assert draft["derived_features"]["bmi"]["value"] == 2000.0
    assert draft["derived_features"]["bmi"]["status"] == "out_of_range"
    assert draft["can_confirm"] is False


def test_categories_are_canonicalized_but_optional_sex_is_never_inferred():
    draft = build_profile_draft(
        [
            candidate("smoking status", "Former smoker", None, "former smoker"),
            candidate("alcohol frequency", "once or twice a week", None, "once or twice a week"),
            candidate("family history", "yes", None, "family history yes"),
        ]
    )

    assert draft["reported_features"]["smoking_status"]["normalized_value"] == "former"
    assert draft["reported_features"]["alcohol_frequency"]["normalized_value"] == "one_to_two_per_week"
    assert draft["reported_features"]["affected_relative"]["normalized_value"] is True
    assert "sex" not in draft["reported_features"]


def test_remove_operation_clears_only_the_named_reported_feature():
    draft = build_profile_draft(
        [
            candidate("age", 55, None, "age 55"),
            candidate("weight", 80, "kg", "weight 80 kg"),
            candidate("age", None, None, "remove age", "remove"),
        ]
    )

    assert "age" not in draft["reported_features"]
    assert draft["reported_features"]["weight"]["normalized_value"] == 80


def test_profile_rate_limit_is_deterministic_per_client_window():
    limiter = ProfileRateLimiter(limit=2, window_seconds=60)

    assert limiter.allow("client-a", now=0) is True
    assert limiter.allow("client-a", now=1) is True
    assert limiter.allow("client-a", now=2) is False
    assert limiter.allow("client-b", now=2) is True
    assert limiter.allow("client-a", now=61) is True


def test_operational_input_bounds_contain_the_current_reference_support():
    for field, (support_min, support_max) in REFERENCE_SUPPORT.items():
        input_min, input_max = INPUT_BOUNDS[field]
        assert input_min <= support_min <= support_max <= input_max


@pytest.mark.asyncio
async def test_extraction_returns_candidates_only_and_preserves_exact_source_text(api_client):
    response_json = {
        "candidates": [
            candidate("height", 180, "cm", "height is 180 cm"),
            candidate("weight", 81, "kg", "weight is 81 kg"),
        ]
    }
    with patch("app.client") as llm:
        llm.chat.completions.create.return_value.choices[0].message.content = json.dumps(
            response_json
        )
        response = await api_client.post(
            "/profile/extract",
            json={"message": "My synthetic profile: height is 180 cm and weight is 81 kg"},
        )

    assert response.status_code == 200
    body = response.json()
    assert list(body) == ["candidates"]
    assert body["candidates"][0]["source_text"] == "height is 180 cm"
    assert "confirmed" not in json.dumps(body).lower()
    assert "match" not in json.dumps(body).lower()


@pytest.mark.asyncio
async def test_extraction_requests_json_output_for_a_message_without_candidates(api_client):
    with patch("app.client") as llm:
        llm.chat.completions.create.return_value.choices[0].message.content = json.dumps(
            {"candidates": []}
        )

        response = await api_client.post(
            "/profile/extract",
            json={"message": "predict"},
        )

    assert response.status_code == 200
    assert response.json() == {"candidates": []}
    request = llm.chat.completions.create.call_args.kwargs
    assert request["response_format"] == {"type": "json_object"}
    assert '{"candidates": []}' in request["messages"][0]["content"]


@pytest.mark.asyncio
async def test_profile_contract_validates_then_requires_a_separate_confirm_call(api_client):
    candidates = [
        candidate("height", 180, "cm", "height 180 cm"),
        candidate("weight", 81, "kg", "weight 81 kg"),
    ]

    validation = await api_client.post("/profile/validate", json={"candidates": candidates})
    assert validation.status_code == 200
    draft = validation.json()
    assert draft["state"] == "draft"
    assert draft["derived_features"]["bmi"]["value"] == 25.0
    assert "matching_started" not in draft

    confirmation = await api_client.post("/profile/confirm", json={"draft": draft})
    assert confirmation.status_code == 200
    assert confirmation.json()["state"] == "confirmed"
    assert confirmation.json()["matching_started"] is False


@pytest.mark.asyncio
async def test_confirm_revalidates_candidates_and_rejects_a_tampered_draft(api_client):
    draft = build_profile_draft([candidate("weight", 80, None, "weight 80")])
    draft["can_confirm"] = True
    draft["reported_features"]["weight"]["status"] = "valid"

    response = await api_client.post("/profile/confirm", json={"draft": draft})

    assert response.status_code == 409
    assert "not eligible" in response.json()["detail"]


@pytest.mark.asyncio
async def test_extraction_has_bounded_input(api_client):
    oversized = await api_client.post(
        "/profile/extract",
        json={"message": "x" * 2001},
    )

    assert oversized.status_code == 422


def test_invalid_candidate_operation_is_blocking_not_reinterpreted_as_set():
    draft = build_profile_draft(
        [candidate("age", 55, "years", "age 55", "invented-operation")]
    )

    assert draft["reported_features"]["age"]["status"] == "ambiguous"
    assert draft["can_confirm"] is False


def test_extraction_contract_rejects_markdown_wrapped_json():
    with pytest.raises(ValueError, match="valid JSON"):
        parse_feature_candidates("age 55", '```json\n{"candidates": []}\n```')


@pytest.mark.asyncio
async def test_validation_contract_rejects_an_unknown_operation(api_client):
    invalid = candidate("age", 55, "years", "age 55", "invented-operation")

    response = await api_client.post("/profile/validate", json={"candidates": [invalid]})

    assert response.status_code == 422

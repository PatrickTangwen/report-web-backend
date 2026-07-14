import csv
from pathlib import Path

import pytest

from icd_keyword import load_icd_vocabulary, match_icd_keywords, validate_vocabulary


ROOT = Path(__file__).resolve().parents[1]


def embedding_codes():
    path = ROOT / "viz" / "data" / "icd_code_embeddings.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [row["code"] for row in csv.DictReader(handle)]


def test_reviewed_vocabulary_selectors_exist_in_the_tracked_icd_embedding():
    vocabulary = load_icd_vocabulary()

    assert vocabulary["version"] == "2026-07-13.v1"
    assert vocabulary["icd_version"] == "ICD-10 2019"
    assert vocabulary["source"]["url"].startswith("https://icd.who.int/")
    assert {entry["selector"]["type"] for entry in vocabulary["entries"]} == {
        "exact",
        "prefix",
        "range",
    }
    assert validate_vocabulary(vocabulary, embedding_codes()) == []


def test_deterministic_mapping_keeps_multiple_icd_matches_separate():
    result = match_icd_keywords("Show tuberculosis and CKD on the ICD graph")

    assert result["status"] == "supported"
    assert result["vocabulary_version"] == "2026-07-13.v1"
    assert [match["id"] for match in result["matches"]] == [
        "tuberculosis",
        "chronic-kidney-disease",
    ]
    assert result["matches"][0]["selector"] == {
        "type": "range",
        "start": "A15",
        "end": "A19",
    }
    assert "profile" not in str(result).lower()
    assert "patient" not in str(result).lower()


def test_more_specific_reviewed_phrase_wins_without_guessing():
    result = match_icd_keywords("Find type 2 diabetes")

    assert result["status"] == "supported"
    assert [match["id"] for match in result["matches"]] == ["type-2-diabetes"]


def test_ambiguous_and_unsupported_input_never_produce_actions():
    ambiguous_vocabulary = {
        "version": "test",
        "icd_version": "ICD-10 2019",
        "ambiguities": [
            {"term": "shared term", "entry_ids": ["left", "right"]},
        ],
        "entries": [
            {
                "id": "left",
                "canonical_keyword": "shared term",
                "aliases": [],
                "display_label": "Left",
                "selector": {"type": "exact", "code": "A00"},
                "selector_label": "A00",
                "source": "reviewed test fixture",
                "icd_version": "ICD-10 2019",
            },
            {
                "id": "right",
                "canonical_keyword": "shared term",
                "aliases": [],
                "display_label": "Right",
                "selector": {"type": "exact", "code": "A01"},
                "selector_label": "A01",
                "source": "reviewed test fixture",
                "icd_version": "ICD-10 2019",
            },
        ],
    }

    assert validate_vocabulary(ambiguous_vocabulary, embedding_codes()) == []
    ambiguous = match_icd_keywords("shared term", ambiguous_vocabulary)
    unsupported = match_icd_keywords("eczema")

    assert ambiguous["status"] == "ambiguous"
    assert ambiguous["matches"] == []
    assert [option["id"] for option in ambiguous["ambiguities"][0]["options"]] == [
        "left",
        "right",
    ]
    assert unsupported == {
        "status": "unsupported",
        "vocabulary_version": "2026-07-13.v1",
        "matches": [],
        "ambiguities": [],
    }


def test_unreviewed_duplicate_terms_remain_vocabulary_errors():
    vocabulary = load_icd_vocabulary()
    duplicate = dict(vocabulary["entries"][0])
    duplicate["id"] = "unreviewed-duplicate"
    vocabulary = {**vocabulary, "entries": [*vocabulary["entries"], duplicate]}

    assert any(
        "is also assigned" in error
        for error in validate_vocabulary(vocabulary, embedding_codes())
    )
    with pytest.raises(ValueError, match="ambiguous without a reviewed declaration"):
        match_icd_keywords(vocabulary["entries"][0]["canonical_keyword"], vocabulary)

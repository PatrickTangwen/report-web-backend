import os
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ["LLM_Key_Deepseek"] = "test-key"

from app import (
    app,
    PAPER_QA_SYSTEM_PROMPT,
    CLINICAL_PROMPT,
    DATA_QUERY_SYSTEM_PROMPT,
    classify_intent,
)
from paper_context import PAPER_TEXT
from data_query import (
    format_data_context,
    is_embedding_query,
    is_feature_importance_query,
    is_pathway_query,
    match_disease,
    match_datasets,
    query_data,
)
from followup import get_pathway_enrichment, describe_embedding_context


def _mock_response(content):
    msg = MagicMock()
    msg.message.content = content
    resp = MagicMock()
    resp.choices = [msg]
    return resp


@pytest.fixture
def mock_openai():
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "This is a test response."
        )
        yield mock_client


@pytest.fixture
def mock_openai_with_intent():
    """Mock that returns intent on first call and answer on second call."""
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("paper_qa"),
            _mock_response("The average AUC is 0.76."),
        ]
        yield mock_client


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- Health ---

@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_superseded_risk_routes_are_not_public(client):
    assert (await client.get("/form-fields", params={"disease": "CKD"})).status_code == 404
    assert (
        await client.post(
            "/clinical/submit",
            json={"disease": "CKD", "values": {"BMI": 25}},
        )
    ).status_code == 404
    assert (
        await client.post(
            "/assess",
            json={"disease": "CKD", "values": {"BMI": 25}},
        )
    ).status_code == 404


@pytest.mark.asyncio
async def test_public_pages_origin_is_allowed_by_cors(client):
    response = await client.options(
        "/chat",
        headers={
            "Origin": "https://patricktangwen.github.io",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://patricktangwen.github.io"


@pytest.mark.asyncio
async def test_local_preview_origins_are_allowed_by_cors(client):
    for origin in (
        "http://localhost:4200",
        "http://localhost:4210",
        "http://127.0.0.1:4999",
        "http://[::1]:4200",
    ):
        response = await client.options(
            "/chat",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )

        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == origin


def test_superseded_mock_and_duplicate_patient_assets_are_absent():
    root = os.path.dirname(os.path.dirname(__file__))
    legacy_paths = [
        "chatbot-backend/data/feature_importance.csv",
        "chatbot-backend/data/fibrotic_patient_embeddings.csv",
        "viz/data/feature_importance.csv",
        "viz/data/fibrotic_patient_embeddings.csv",
    ]

    assert [path for path in legacy_paths if os.path.exists(os.path.join(root, path))] == []


# --- Paper Q&A ---

@pytest.mark.asyncio
async def test_chat_paper_qa(client, mock_openai_with_intent):
    resp = await client.post(
        "/chat", json={"message": "What is the average AUC?"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "paper_qa"
    assert "0.76" in body["reply"]

    qa_call = mock_openai_with_intent.chat.completions.create.call_args_list[1]
    messages = qa_call.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "PAPER CONTENT" in messages[0]["content"]


@pytest.mark.asyncio
async def test_chat_with_history(client, mock_openai_with_intent):
    resp = await client.post(
        "/chat",
        json={
            "message": "Follow up question",
            "history": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First reply"},
            ],
        },
    )
    assert resp.status_code == 200
    qa_call = mock_openai_with_intent.chat.completions.create.call_args_list[1]
    messages = qa_call.kwargs["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[3]["content"] == "Follow up question"


# --- Intent classification ---

@pytest.mark.asyncio
async def test_intent_clinical_offers_the_research_demo_profile(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response("clinical")
        resp = await client.post(
            "/chat", json={"message": "Assess my risk for heart disease"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "clinical"
        assert body["reply"] == CLINICAL_PROMPT
        assert body["ui"] == {"type": "demo_profile_start"}
        assert "not medical advice" in body["reply"]
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_intent_data_query_routes_to_data(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("data_query"),
            _mock_response("The AUROC for CKD is 0.82."),
        ]
        resp = await client.post(
            "/chat", json={"message": "What's the AUROC for CKD?"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "data_query"
        assert body["reply"] == "The AUROC for CKD is 0.82."

        qa_call = mock_client.chat.completions.create.call_args_list[1]
        system_content = qa_call.kwargs["messages"][0]["content"]
        assert "QUERY RESULTS" in system_content


@pytest.mark.asyncio
async def test_intent_general_falls_through_to_paper_qa(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("general"),
            _mock_response("Hello! I can help you with questions about the paper."),
        ]
        resp = await client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "general"


@pytest.mark.asyncio
async def test_intent_unknown_defaults_to_paper_qa(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("something_weird!!!"),
            _mock_response("Answering from paper context."),
        ]
        resp = await client.post(
            "/chat", json={"message": "Tell me about the model"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "paper_qa"


def test_classify_intent_sanitizes_output():
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "  Paper_QA  "
        )
        result = classify_intent("What is ALIGATEHR-Gen?")
        assert result == "paper_qa"


def test_classify_intent_garbage_defaults():
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response(
            "I think this is about the paper"
        )
        result = classify_intent("What is the AUC?")
        assert result == "paper_qa"


# --- ICD keyword walkthrough ---

@pytest.mark.asyncio
async def test_chat_returns_separate_deterministic_icd_actions_without_llm(client):
    with patch("app.client") as mock_client:
        resp = await client.post(
            "/chat",
            json={"message": "Show tuberculosis and CKD on the ICD graph"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "icd_keyword_match"
    assert body["ui"]["type"] == "icd_keyword_matches"
    assert [match["id"] for match in body["ui"]["matches"]] == [
        "tuberculosis",
        "chronic-kidney-disease",
    ]
    assert set(body["ui"]["matches"][0]) == {
        "id",
        "canonical_keyword",
        "matched_keyword",
        "display_label",
        "selector",
        "selector_label",
    }
    assert "not patient history or clinical similarity" in body["reply"]
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_explicit_unsupported_icd_request_is_not_guessed(client):
    with patch("app.client") as mock_client:
        resp = await client.post(
            "/chat", json={"message": "Highlight eczema on the ICD graph"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "icd_keyword_match"
    assert body["ui"] == {
        "type": "icd_keyword_unsupported",
        "vocabulary_version": "2026-07-13.v1",
    }
    assert "not in the reviewed ICD Keyword Vocabulary" in body["reply"]
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_plain_disease_question_keeps_existing_intent_routing(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("paper_qa"),
            _mock_response("Paper answer."),
        ]
        resp = await client.post(
            "/chat", json={"message": "What does the paper say about CKD?"}
        )

    assert resp.status_code == 200
    assert resp.json()["intent"] == "paper_qa"
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_unrelated_highlight_request_keeps_existing_intent_routing(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("paper_qa"),
            _mock_response("Paper answer."),
        ]
        resp = await client.post(
            "/chat", json={"message": "Highlight the paper's main contribution"}
        )

    assert resp.status_code == 200
    assert resp.json()["intent"] == "paper_qa"
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_paper_risk_factor_question_keeps_paper_routing(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("paper_qa"),
            _mock_response("Paper answer."),
        ]
        response = await client.post(
            "/chat",
            json={"message": "What risk factors does the paper discuss?"},
        )

    assert response.status_code == 200
    assert response.json()["intent"] == "paper_qa"
    assert response.json()["reply"] == "Paper answer."
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_unranked_project_risk_factor_question_keeps_research_routing(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("paper_qa"),
            _mock_response("Research answer."),
        ]
        response = await client.post(
            "/chat",
            json={"message": "What risk factors does ALIGATEHR-Gen discuss?"},
        )

    assert response.status_code == 200
    assert response.json()["intent"] == "paper_qa"
    assert response.json()["reply"] == "Research answer."
    assert mock_client.chat.completions.create.call_count == 2


@pytest.mark.asyncio
async def test_ranked_paper_risk_factor_question_keeps_paper_routing(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("paper_qa"),
            _mock_response("Paper answer."),
        ]
        response = await client.post(
            "/chat",
            json={
                "message": "What are the most important risk factors discussed in the paper?"
            },
        )

    assert response.status_code == 200
    assert response.json()["intent"] == "paper_qa"
    assert response.json()["reply"] == "Paper answer."


@pytest.mark.asyncio
async def test_basic_reviewed_keyword_action_does_not_require_graph_jargon(client):
    with patch("app.client") as mock_client:
        resp = await client.post(
            "/chat", json={"message": "Find type 2 diabetes"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["intent"] == "icd_keyword_match"
    assert [match["id"] for match in body["ui"]["matches"]] == [
        "type-2-diabetes"
    ]
    mock_client.chat.completions.create.assert_not_called()


# --- Data query module ---

def test_match_disease_exact():
    assert match_disease("What's the AUROC for CKD?") == "CKD"


def test_match_disease_full_name():
    assert match_disease("Tell me about Type 2 Diabetes") == "Type 2 Diabetes"


def test_match_disease_underscore():
    assert match_disease("pathways for Crohns Disease") == "Crohns_Disease"


def test_match_disease_alias_mash():
    assert match_disease("Top risk factors for MASH?") == "NASH"


def test_match_disease_alias_cad():
    assert match_disease("What about coronary artery disease?") == "Coronary Heart Disease"


def test_match_disease_alias_crohn():
    assert match_disease("pathways for Crohn's disease") == "Crohns_Disease"


def test_match_disease_none():
    assert match_disease("What's the best model?") is None


def test_match_disease_case_insensitive():
    result = match_disease("what about ckd?")
    assert result == "CKD"


def test_match_datasets_eval_keywords():
    result = match_datasets("What's the AUROC for CKD?")
    assert "evaluation_metrics" in result


def test_match_datasets_ablation_keywords():
    result = match_datasets("What happens without genetic data?")
    assert "ablation_results" in result


def test_match_datasets_feature_keywords():
    query = "What are the top risk factors for MASH?"
    assert is_feature_importance_query(query)
    assert "feature_importance" not in match_datasets(query)


def test_match_datasets_pathway_keywords():
    result = match_datasets("What pathways are enriched in CKD?")
    assert "pathway_enrichment" in result


def test_match_datasets_defaults_to_eval():
    result = match_datasets("Tell me something about the data")
    assert result == ["evaluation_metrics"]


def test_query_data_filters_by_disease():
    results, disease, names = query_data("What's the AUROC for CKD?")
    assert disease == "CKD"
    assert "evaluation_metrics" in results
    assert "CKD" in results["evaluation_metrics"]
    assert "Type 2 Diabetes" not in results["evaluation_metrics"]


def test_query_data_ablation():
    results, disease, names = query_data("What happens without genetic data?")
    assert "ablation_results" in results
    assert "w/o Genetic Data" in results["ablation_results"]


def test_query_data_feature_importance():
    results, disease, names = query_data("Top risk factors for NASH?")
    assert "feature_importance" not in results
    assert "feature_importance" not in names


def test_query_data_pathway():
    results, disease, names = query_data("What pathways are enriched in CKD?")
    assert "pathway_enrichment" in results
    assert "CKD" in results["pathway_enrichment"]


def test_format_data_context_structure():
    results, disease, names = query_data("AUROC for CKD")
    ctx = format_data_context(results, disease, names)
    assert "Available datasets:" in ctx
    assert "Detected disease filter: CKD" in ctx
    assert "=== evaluation_metrics ===" in ctx
    assert "Available diseases per dataset" in ctx


# --- Error handling ---

@pytest.mark.asyncio
async def test_chat_no_api_key(client):
    with patch("app.client", None):
        resp = await client.post("/chat", json={"message": "Hello"})
        assert resp.status_code == 503


@pytest.mark.asyncio
async def test_chat_missing_message_field(client):
    resp = await client.post("/chat", json={})
    assert resp.status_code == 422


# --- Paper context ---

def test_paper_context_contains_key_facts():
    assert "ALIGATEHR-Gen" in PAPER_TEXT
    assert "118 diseases" in PAPER_TEXT
    assert "0.76" in PAPER_TEXT
    assert "UK Biobank" in PAPER_TEXT
    assert "first-degree relatives" in PAPER_TEXT


def test_system_prompt_includes_paper():
    assert "PAPER CONTENT" in PAPER_QA_SYSTEM_PROMPT
    assert "ALIGATEHR-Gen" in PAPER_QA_SYSTEM_PROMPT
    assert "do not guess or hallucinate" in PAPER_QA_SYSTEM_PROMPT


# --- Response model ---

@pytest.mark.asyncio
async def test_response_includes_intent_field(client, mock_openai_with_intent):
    resp = await client.post(
        "/chat", json={"message": "How does the attention mechanism work?"}
    )
    body = resp.json()
    assert "intent" in body
    assert "reply" in body



# --- Followup: pathway enrichment ---

def test_get_pathway_enrichment_valid():
    result = get_pathway_enrichment("CKD")
    assert result is not None
    assert result["disease"] == "CKD"
    assert "Chronic Kidney Disease" in result["disease_label"]
    assert len(result["pathways"]) == 10
    first = result["pathways"][0]
    assert "pathway" in first
    assert "source" in first
    assert "gene_count" in first
    assert "enrichment_ratio" in first
    assert "p_adjusted" in first


def test_get_pathway_enrichment_top_n():
    result = get_pathway_enrichment("CKD", top_n=5)
    assert result is not None
    assert len(result["pathways"]) == 5


def test_get_pathway_enrichment_alias():
    result = get_pathway_enrichment("mash")
    assert result is not None
    assert result["disease"] == "NASH"


def test_get_pathway_enrichment_unknown():
    result = get_pathway_enrichment("nonexistent_xyz")
    assert result is None


def test_get_pathway_enrichment_sources():
    result = get_pathway_enrichment("CKD", top_n=40)
    sources = {p["source"] for p in result["pathways"]}
    assert len(sources) >= 2


# --- Followup: embedding context ---

def test_describe_embedding_context_valid():
    result = describe_embedding_context("CKD")
    assert result is not None
    assert result["disease"] == "CKD"
    assert result["embed_disease"] == "CKD"
    assert result["total_patients"] > 0
    assert result["dataset_version"].startswith("fibrotic-")
    assert "x" in result["centroid"]
    assert "y" in result["centroid"]
    assert "mean_purity_2d" not in result
    groups = result["groups"]
    for g in ("pure", "intermediate", "overlap"):
        assert g in groups
        assert "count" in groups[g]
        assert "pct" in groups[g]
    assert len(result["nearest_clusters"]) == 3


def test_describe_embedding_context_alias():
    result = describe_embedding_context("mash")
    assert result is not None
    assert result["disease"] == "NASH"
    assert result["embed_disease"] == "MASH"


def test_describe_embedding_context_unknown():
    result = describe_embedding_context("nonexistent_xyz")
    assert result is None


def test_describe_embedding_context_no_embedding():
    result = describe_embedding_context("IPF")
    assert result is None


def test_describe_embedding_context_group_pcts_sum():
    result = describe_embedding_context("CKD")
    total_pct = sum(result["groups"][g]["pct"] for g in ("pure", "intermediate", "overlap"))
    assert 99.5 <= total_pct <= 100.5


# --- Keyword detection ---

def test_is_pathway_query():
    assert is_pathway_query("What pathways are involved?")
    assert is_pathway_query("Show me the enriched pathways")
    assert is_pathway_query("KEGG pathways for this disease")
    assert not is_pathway_query("What's the AUROC?")


def test_is_embedding_query():
    assert is_embedding_query("Where do similar patients cluster?")
    assert is_embedding_query("Show me the UMAP embedding")
    assert is_embedding_query("What about patient clustering?")
    assert not is_embedding_query("What's the AUROC?")


# --- Chat data queries with explicit disease context ---

@pytest.mark.asyncio
async def test_chat_pathway_disease_in_message(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response("data_query")
        resp = await client.post(
            "/chat",
            json={"message": "What pathways are enriched in CKD?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ui"]["type"] == "pathway_enrichment"
        assert body["ui"]["disease"] == "CKD"


@pytest.mark.asyncio
async def test_chat_embedding_uses_the_authoritative_release(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = [
            _mock_response("data_query"),
            _mock_response("The CKD display group is visible in the t-SNE view."),
        ]
        response = await client.post(
            "/chat",
            json={"message": "Where are CKD patients in the embedding?"},
        )

    assert response.status_code == 200
    system = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"][0][
        "content"
    ]
    assert "Dataset Release:" in system
    assert "Display-only t-SNE centroid" in system
    assert "purity" not in system.lower()


@pytest.mark.asyncio
async def test_chat_refuses_the_superseded_mock_top_10(client):
    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.return_value = _mock_response("data_query")
        response = await client.post(
            "/chat",
            json={"message": "What are the top risk factors for MASH?"},
        )

    assert response.status_code == 200
    assert "superseded mock ranking" in response.json()["reply"]
    mock_client.chat.completions.create.assert_not_called()


@pytest.mark.asyncio
async def test_chat_refuses_mock_top_10_before_icd_keyword_routing(client):
    with patch("app.client", None):
        response = await client.post(
            "/chat",
            json={"message": "Show the Top 10 risk factors for CKD"},
        )

    assert response.status_code == 200
    assert response.json()["intent"] == "data_query"
    assert response.json()["ui"] is None
    assert "superseded mock ranking" in response.json()["reply"]

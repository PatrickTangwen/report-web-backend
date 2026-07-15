import os
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

os.environ["LLM_Key_Deepseek"] = "test-key"

from app import app, PAPER_QA_SYSTEM_PROMPT


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
            "The model achieves an average AUC of 0.76."
        )
        yield mock_client


@pytest_asyncio.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_paper_question_answers_from_the_paper_only_prompt(client, mock_openai):
    resp = await client.post(
        "/paper/question", json={"message": "What is the average AUC?"}
    )
    assert resp.status_code == 200
    assert resp.json() == {"reply": "The model achieves an average AUC of 0.76."}

    mock_openai.chat.completions.create.assert_called_once()
    call = mock_openai.chat.completions.create.call_args
    assert call.kwargs["messages"][0] == {
        "role": "system",
        "content": PAPER_QA_SYSTEM_PROMPT,
    }
    assert call.kwargs["messages"][-1] == {
        "role": "user",
        "content": "What is the average AUC?",
    }


@pytest.mark.asyncio
async def test_paper_question_carries_conversation_history(client, mock_openai):
    resp = await client.post(
        "/paper/question",
        json={
            "message": "And for the ablation study?",
            "history": [
                {"role": "user", "content": "What is the average AUC?"},
                {"role": "assistant", "content": "0.76."},
            ],
        },
    )
    assert resp.status_code == 200
    messages = mock_openai.chat.completions.create.call_args.kwargs["messages"]
    assert len(messages) == 4
    assert messages[1] == {"role": "user", "content": "What is the average AUC?"}
    assert messages[2] == {"role": "assistant", "content": "0.76."}
    assert messages[3]["content"] == "And for the ablation study?"


@pytest.mark.asyncio
async def test_paper_question_bypasses_intent_classification_and_icd_routing(
    client, mock_openai
):
    """A message that /chat would route to ICD keyword matching (no LLM call at
    all) must still reach the paper-only LLM prompt exactly once here — this is
    the explicit bypass of generic intent classification issue #38 requires."""
    resp = await client.post(
        "/paper/question",
        json={"message": "Show me tuberculosis on the ICD graph"},
    )
    assert resp.status_code == 200
    mock_openai.chat.completions.create.assert_called_once()
    call = mock_openai.chat.completions.create.call_args
    assert call.kwargs["messages"][0]["content"] == PAPER_QA_SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_paper_question_requires_llm_configured(client):
    with patch("app.client", None):
        resp = await client.post("/paper/question", json={"message": "Hello"})
        assert resp.status_code == 503


@pytest.mark.asyncio
async def test_paper_question_rejects_empty_message(client):
    resp = await client.post("/paper/question", json={"message": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_paper_question_rejects_missing_message(client):
    resp = await client.post("/paper/question", json={})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_paper_question_surfaces_llm_errors(client):
    from openai import APIError

    with patch("app.client") as mock_client:
        mock_client.chat.completions.create.side_effect = APIError(
            "boom", request=MagicMock(), body=None
        )
        resp = await client.post("/paper/question", json={"message": "Hello"})
        assert resp.status_code == 502


def test_system_prompt_forbids_external_sources_and_section_links():
    assert "external sources" in PAPER_QA_SYSTEM_PROMPT
    assert "section" in PAPER_QA_SYSTEM_PROMPT

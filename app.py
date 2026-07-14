import os
import re
from contextlib import asynccontextmanager
from typing import Literal

from openai import OpenAI, APIError
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from paper_context import PAPER_TEXT
from data_query import (
    format_data_context,
    is_embedding_query,
    is_feature_importance_query,
    is_pathway_query,
    match_disease,
    query_data,
)
from followup import get_pathway_enrichment, describe_embedding_context
from fibrotic_release import load_fibrotic_release
from demo_profile import (
    FEATURE_EXTRACTION_SYSTEM_PROMPT,
    ProfileRateLimiter,
    build_profile_draft,
    confirm_profile,
    parse_feature_candidates,
)
from profile_matching import load_matching_release, match_confirmed_profile
from icd_keyword import is_icd_keyword_request, match_icd_keywords

INTENT_SYSTEM_PROMPT = (
    "You are an intent classifier. Given a user message, classify it into exactly one "
    "of these categories:\n\n"
    "- data_query: asking for specific metric values, scores, rankings, comparisons "
    "between models, ablation results, feature importance, risk factors, pathway "
    "enrichment, or any quantitative lookup. Examples: "
    '"What\'s the AUROC for CKD?", "Which model performs best on diabetes?", '
    '"What happens without genetic data?", "Top risk factors for MASH?", '
    '"What pathways are enriched in CKD?"\n'
    "- clinical: requests for personal clinical risk assessment, patient-specific "
    "predictions, or guided clinical support for the user themselves\n"
    "- paper_qa: questions about how the model works, the methodology, architecture, "
    "training process, graph construction, attention mechanism, or general discussion "
    "of the paper's contributions and limitations\n"
    "- general: greetings, off-topic, or anything that does not fit the above\n\n"
    "If the user asks for a specific number, score, comparison, or ranking, "
    "classify as data_query.\n\n"
    "Respond with ONLY the category name, nothing else."
)

PAPER_QA_SYSTEM_PROMPT = (
    "You are a research assistant for the ALIGATEHR-Gen project. "
    "Answer questions using ONLY the paper content provided below. "
    "If the answer is not in the paper, say so explicitly — do not guess or hallucinate.\n\n"
    "Keep answers concise, scientifically accurate, and well-structured. "
    "Use specific numbers and facts from the paper when relevant.\n\n"
    "Important: This is a research prototype for demonstration purposes only. "
    "Any clinical information discussed should not be used for medical decision-making.\n\n"
    "--- PAPER CONTENT ---\n"
    f"{PAPER_TEXT}\n"
    "--- END PAPER CONTENT ---"
)

CLINICAL_PROMPT = (
    "I can help you build a synthetic or sufficiently de-identified Demo Profile "
    "for a research comparison. This is not medical advice, diagnosis, or a personal "
    "outcome prediction. Do not enter names, exact birth dates, addresses, patient "
    "IDs, or real medical records."
)

DATA_QUERY_SYSTEM_PROMPT = (
    "You are a data analyst for the ALIGATEHR-Gen project. "
    "Answer questions using ONLY the query results provided below. "
    "If the data does not contain the answer, say so explicitly.\n\n"
    "Present numbers precisely as they appear in the data. "
    "Use tables or bullet points for comparative answers. "
    "Keep answers concise and well-structured.\n\n"
    "Important: This is a research prototype for demonstration purposes only.\n\n"
)

INTENT_LABELS = ("paper_qa", "clinical", "data_query", "general")


def _format_embedding_context(emb):
    groups = emb["groups"]
    lines = [
        f"Disease: {emb['disease_label']} ({emb['embed_disease']})",
        f"Dataset Release: {emb['dataset_version']}",
        f"Total patients in embedding space: {emb['total_patients']}",
        f"Display-only t-SNE centroid: ({emb['centroid']['x']}, {emb['centroid']['y']})",
        "",
        "Patient group distribution:",
        f"  Pure: {groups['pure']['count']} ({groups['pure']['pct']}%)",
        f"  Intermediate: {groups['intermediate']['count']} ({groups['intermediate']['pct']}%)",
        f"  Overlap: {groups['overlap']['count']} ({groups['overlap']['pct']}%)",
        "",
        "Nearest disease display groups (by t-SNE centroid distance):",
    ]
    for n in emb["nearest_clusters"]:
        lines.append(f"  {n['disease']}: distance {n['distance']}")
    return "\n".join(lines)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = []


class ChatResponse(BaseModel):
    reply: str
    intent: str = "paper_qa"
    ui: dict | None = None


class FeatureCandidate(BaseModel):
    field: str = Field(min_length=1, max_length=80)
    raw_value: str | float | int | bool | None
    raw_unit: str | None = Field(default=None, max_length=40)
    source_text: str = Field(min_length=1, max_length=500)
    operation: Literal["set", "correct", "remove"] = "set"


class ProfileExtractRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ProfileExtractResponse(BaseModel):
    candidates: list[FeatureCandidate]


class ProfileValidateRequest(BaseModel):
    candidates: list[FeatureCandidate] = Field(max_length=100)


class ProfileDraftInput(BaseModel):
    state: str
    candidates: list[FeatureCandidate] = Field(max_length=100)


class ProfileConfirmRequest(BaseModel):
    draft: ProfileDraftInput


class ProfileMatchRequest(BaseModel):
    confirmed_profile: ProfileDraftInput
    target: Literal[
        "CKD",
        "Cardiac_Fibrosis",
        "MASH",
        "Pulmonary_fibrosis",
        "SSc_Connective_Tissue",
        "Crohns_Disease",
        "Fibrosis_of_Skin",
    ]


client: OpenAI | None = None
profile_rate_limiter = ProfileRateLimiter()


def get_profile_matching_release():
    try:
        return load_matching_release()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise HTTPException(
            status_code=503,
            detail=f"Profile matching release unavailable: {error}",
        )


def enforce_profile_rate_limit(request):
    client_key = request.client.host if request.client else "unknown"
    if not profile_rate_limiter.allow(client_key):
        raise HTTPException(status_code=429, detail="Too many profile requests")


@asynccontextmanager
async def lifespan(app):
    global client
    api_key = os.environ.get("LLM_Key_Deepseek")
    if api_key:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    yield


app = FastAPI(title="ALIGATEHR-Gen Chatbot", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://patricktangwen.github.io",
        "https://patirckistc-report-web.hf.space",
    ],
    allow_origin_regex=r"^http://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def classify_intent(user_message):
    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=16,
        temperature=0,
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    raw = response.choices[0].message.content.strip().lower()
    raw = re.sub(r"[^a-z_]", "", raw)
    if raw in INTENT_LABELS:
        return raw
    return "paper_qa"


@app.get("/health")
async def health():
    get_profile_matching_release()
    return {"status": "ok"}


def _release_headers(dataset_version):
    return {
        "Cache-Control": "public, max-age=300",
        "ETag": f'"{dataset_version}"',
    }


@app.get("/embedding/fibrotic")
async def fibrotic_embedding(request: Request, response: Response):
    embedding, _ = load_fibrotic_release()
    headers = _release_headers(embedding["dataset_version"])
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    response.headers.update(headers)
    return embedding


@app.get("/embedding/fibrotic/preset")
async def fibrotic_preset(response: Response):
    embedding, preset = load_fibrotic_release()
    response.headers.update(_release_headers(embedding["dataset_version"]))
    return preset


@app.post("/profile/extract", response_model=ProfileExtractResponse)
async def profile_extract(req: ProfileExtractRequest, request: Request):
    enforce_profile_rate_limit(request)
    if client is None:
        raise HTTPException(status_code=503, detail="LLM API key not configured")
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=900,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FEATURE_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": req.message},
            ],
        )
        candidates = parse_feature_candidates(
            req.message, response.choices[0].message.content
        )
    except APIError as error:
        raise HTTPException(status_code=502, detail=f"LLM API error: {error.message}")
    except (AttributeError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error))
    return {"candidates": candidates}


@app.post("/profile/validate")
async def profile_validate(req: ProfileValidateRequest, request: Request):
    enforce_profile_rate_limit(request)
    return build_profile_draft([candidate.model_dump() for candidate in req.candidates])


@app.post("/profile/confirm")
async def profile_confirm(req: ProfileConfirmRequest, request: Request):
    enforce_profile_rate_limit(request)
    try:
        return confirm_profile(req.draft.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))


@app.post("/profile/match")
async def profile_match(
    req: ProfileMatchRequest,
    request: Request,
    release=Depends(get_profile_matching_release),
):
    enforce_profile_rate_limit(request)
    if req.confirmed_profile.state != "confirmed":
        raise HTTPException(status_code=409, detail="Profile must be explicitly confirmed")
    try:
        confirmed = confirm_profile(req.confirmed_profile.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error))
    return match_confirmed_profile(
        confirmed,
        req.target,
        release["rows"],
        release["calibration"],
        release["dataset_version"],
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if is_feature_importance_query(req.message):
        return ChatResponse(
            reply=(
                "Feature-importance and Top-10 risk-factor results are not "
                "published in this research experience. I will not answer from "
                "the superseded mock ranking."
            ),
            intent="data_query",
        )

    mapping = match_icd_keywords(req.message)
    if is_icd_keyword_request(
        req.message,
        has_reviewed_keyword=mapping["status"] in {"supported", "ambiguous"},
    ):
        if mapping["status"] == "supported":
            count = len(mapping["matches"])
            return ChatResponse(
                reply=(
                    f"I found {count} reviewed ICD Keyword "
                    f"{'Match' if count == 1 else 'Matches'}. "
                    "Each action opens one code selection independently; these are "
                    "navigation aids, not patient history or clinical similarity."
                ),
                intent="icd_keyword_match",
                ui={
                    "type": "icd_keyword_matches",
                    "vocabulary_version": mapping["vocabulary_version"],
                    "matches": mapping["matches"],
                },
            )
        if mapping["status"] == "ambiguous":
            labels = sorted(
                {
                    option["display_label"]
                    for ambiguity in mapping["ambiguities"]
                    for option in ambiguity["options"]
                }
            )
            return ChatResponse(
                reply=(
                    "That keyword is ambiguous in the reviewed vocabulary. Please "
                    f"specify one of: {', '.join(labels)}."
                ),
                intent="icd_keyword_match",
                ui={
                    "type": "icd_keyword_ambiguous",
                    "vocabulary_version": mapping["vocabulary_version"],
                    "ambiguities": mapping["ambiguities"],
                },
            )
        return ChatResponse(
            reply=(
                "That keyword is not in the reviewed ICD Keyword Vocabulary, so I "
                "will not guess a code or range."
            ),
            intent="icd_keyword_match",
            ui={
                "type": "icd_keyword_unsupported",
                "vocabulary_version": mapping["vocabulary_version"],
            },
        )

    if client is None:
        raise HTTPException(status_code=503, detail="LLM API key not configured")

    try:
        intent = classify_intent(req.message)
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")

    if intent == "clinical":
        return ChatResponse(
            reply=CLINICAL_PROMPT,
            intent=intent,
            ui={"type": "demo_profile_start"},
        )

    if intent == "data_query":
        disease_from_msg = match_disease(req.message)
        disease = disease_from_msg

        if is_pathway_query(req.message) and disease:
            pw = get_pathway_enrichment(disease)
            if pw:
                return ChatResponse(
                    reply=f"Here are the top enriched pathways for **{pw['disease_label']}**:",
                    intent="data_query",
                    ui={"type": "pathway_enrichment", **pw},
                )

        if is_embedding_query(req.message) and disease:
            emb = describe_embedding_context(disease)
            if emb:
                emb_context = _format_embedding_context(emb)
                system = DATA_QUERY_SYSTEM_PROMPT + "--- EMBEDDING CONTEXT ---\n" + emb_context + "\n--- END EMBEDDING CONTEXT ---"
                messages = [{"role": "system", "content": system}]
                messages.extend({"role": m.role, "content": m.content} for m in req.history)
                messages.append({"role": "user", "content": req.message})
                try:
                    response = client.chat.completions.create(
                        model="deepseek-chat", max_tokens=1024, messages=messages,
                    )
                    reply = response.choices[0].message.content
                except APIError as e:
                    raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")
                return ChatResponse(reply=reply, intent="data_query")

        results, disease_q, dataset_names = query_data(req.message)
        data_context = format_data_context(results, disease_q, dataset_names)
        system = DATA_QUERY_SYSTEM_PROMPT + "--- QUERY RESULTS ---\n" + data_context + "\n--- END QUERY RESULTS ---"
    else:
        system = PAPER_QA_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system}]
    messages.extend({"role": m.role, "content": m.content} for m in req.history)
    messages.append({"role": "user", "content": req.message})

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1024,
            messages=messages,
        )
        reply = response.choices[0].message.content
    except APIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {e.message}")

    return ChatResponse(reply=reply, intent=intent)

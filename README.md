---
title: ALIGATEHR-Gen Chatbot Backend
emoji: "🧬"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ALIGATEHR-Gen Chatbot Backend

FastAPI backend for the ALIGATEHR-Gen research demonstration. This is a
research prototype and does not provide medical advice, diagnosis, or personal
outcome predictions.

The public deployment source contains only the de-identified visualization
release. The matching artifact is read at runtime from a private Hugging Face
Dataset. Deployment requires these environment settings:

- Variable `FIBROTIC_MATCH_DATASET_REPO`
- Variable `FIBROTIC_MATCH_DATASET_REVISION`
- Secret `HF_TOKEN`, scoped to read only that private Dataset
- Secret `LLM_Key_Deepseek`

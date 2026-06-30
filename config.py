"""Shared configuration for Lab 24: Eval + Guardrail Stack."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
MINI_MAX_API_KEY = os.getenv("MINI_MAX_API_KEY", "")
MINI_MAX_ENDPOINT = os.getenv("MINI_MAX_ENDPOINT", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Optional: for HuggingFace models

# --- Qdrant (same as Day 18) ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_ENDPOINT = os.getenv("QDRANT_ENDPOINT", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "lab24_production"

# --- Embedding (same as Day 18) ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking (same as Day 18) ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search (same as Day 18) ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set_50q.json")
ANSWERS_PATH = os.path.join(os.path.dirname(__file__), "answers_50q.json")
HUMAN_LABELS_PATH = os.path.join(os.path.dirname(__file__), "human_labels_10q.json")
ADVERSARIAL_SET_PATH = os.path.join(os.path.dirname(__file__), "adversarial_set_20.json")
GUARDRAILS_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "guardrails")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")


def _clean_llm_response(text: str) -> str:
    import re
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _wrap_clean_client(client):
    """Strip MiniMax thinking blocks from chat completions."""
    original_create = client.chat.completions.create

    def create(**kwargs):
        resp = original_create(**kwargs)
        if resp.choices:
            msg = resp.choices[0].message
            if msg.content:
                msg.content = _clean_llm_response(msg.content)
        return resp

    client.chat.completions.create = create
    return client


def get_llm_client_and_model():
    """Ưu tiên Groq, sau đó MiniMax, cuối cùng OpenAI."""
    from openai import OpenAI

    if GROQ_API_KEY:
        return OpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL), GROQ_MODEL
    if MINI_MAX_API_KEY and MINI_MAX_ENDPOINT:
        return _wrap_clean_client(
            OpenAI(api_key=MINI_MAX_API_KEY, base_url=MINI_MAX_ENDPOINT)
        ), "MiniMax-M3"
    if OPENAI_API_KEY:
        return OpenAI(api_key=OPENAI_API_KEY), JUDGE_MODEL
    return None, None


def get_qdrant_client():
    """Qdrant cloud (nếu có endpoint) hoặc local docker."""
    from qdrant_client import QdrantClient

    if QDRANT_ENDPOINT:
        return QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY or None, timeout=300)
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)


def generate_answer(query: str, contexts: list[str]) -> str:
    """Generate answer từ retrieved contexts."""
    if not contexts:
        return "Không tìm thấy thông tin."

    client, model = get_llm_client_and_model()
    if not client:
        return contexts[0]

    context_str = "\n\n".join(contexts)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ],
            max_tokens=300,
        )
        return _clean_llm_response(resp.choices[0].message.content.strip())
    except Exception as e:
        print(f"  ⚠️  LLM generation failed: {e}", flush=True)
        return contexts[0]


# --- LLM Judge ---
JUDGE_MODEL = GROQ_MODEL

# --- Guardrail latency budget ---
LATENCY_BUDGET_P95_MS = 500  # target: full guard stack P95 < 500ms
PRESIDIO_LANGUAGE = "en"    # Presidio base language; custom VN recognizers added via PatternRecognizer

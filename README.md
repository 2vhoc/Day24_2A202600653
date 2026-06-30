# Lab 24 — Production Eval + Guardrail Stack

**Sinh viên:** Vũ Văn Học  
**MSSV:** 2A202600653  
**Repository:** [git@github.com:2vhoc/Day24_2A202600653.git](https://github.com/2vhoc/Day24_2A202600653)

---

## Tổng quan

Xây dựng **complete eval + guardrail stack** trên RAG pipeline Day 18:

```
[Day 18 Pipeline]
    ├── Phase A: RAGAS 50q      → Tìm điểm yếu pipeline
    ├── Phase B: LLM-as-Judge   → Đo độ tin cậy của eval
    └── Phase C: NeMo Guardrails → Bảo vệ khỏi input độc hại
```

---

## Kết quả đạt được

| Hạng mục | Kết quả |
|---|---|
| `pytest tests/` | **40/40 passed** |
| Phase A (Tasks 1–4) | ✅ Implemented |
| Phase B (Tasks 5–8) | ✅ Implemented |
| Phase C (Tasks 9–12) | ✅ Implemented |
| Adversarial suite | **20/20** blocked đúng |
| Presidio PII | VN_CCCD, VN_PHONE, EMAIL |
| CI/CD Blueprint | `reports/blueprint.md` |
| Failure analysis | `analysis/failure_clusters.md` |
| Bias analysis | `analysis/bias_report.md` |

### Điểm làm tốt

1. **Guard stack 2 tầng:** Presidio (PII, <12ms P95) + rule-based + NeMo input rail.
2. **Adversarial 20/20:** Rule-based patterns bổ sung NeMo, không phụ thuộc 100% LLM API.
3. **Custom VN PII recognizers:** Không cần `en_core_web_lg`, tránh false positive.
4. **Swap-and-average judge:** Phát hiện position bias, fallback heuristic khi API hết quota.
5. **RAGAS cluster analysis:** Matrix worst_metric × distribution + dominant failure insight.

---

## Cấu trúc thư mục

```
├── src/
│   ├── m1_chunking.py … m5_enrichment.py, pipeline.py  ← Day 18
│   ├── phase_a_ragas.py    ← Tasks 1–4
│   ├── phase_b_judge.py    ← Tasks 5–8
│   └── phase_c_guard.py    ← Tasks 9–12
├── guardrails/             ← NeMo config + rails.co
├── data/                   ← 25 tài liệu HR policy
├── reports/
│   ├── blueprint.md
│   ├── judge_results.json
│   └── guard_results.json
├── analysis/
│   ├── failure_clusters.md
│   └── bias_report.md
├── tests/                  ← 40 unit tests
└── setup_answers.py        ← Generate answers_50q.json
```

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # Điền MINI_MAX_API_KEY hoặc OPENAI_API_KEY + Qdrant

# Generate answers (cần API + Qdrant, ~5–10 phút)
python setup_answers.py

# Chạy từng phase
python src/phase_a_ragas.py
python src/phase_b_judge.py
python src/phase_c_guard.py

# Kiểm tra
pytest tests/ -v
python check_lab.py
```

### Biến môi trường (`.env`)

| Biến | Mục đích |
|---|---|
| `MINI_MAX_API_KEY` + `MINI_MAX_ENDPOINT` | LLM judge + RAGAS (ưu tiên) |
| `OPENAI_API_KEY` | Fallback cho NeMo guardrails |
| `QDRANT_ENDPOINT` + `QDRANT_API_KEY` | Vector DB cho pipeline |

---

## Các Phase

### Phase A — RAGAS (50 câu, 3 distributions)

| Distribution | Số câu | Đặc điểm |
|---|---|---|
| factual | 20 | Tra cứu đơn giản |
| multi_hop | 20 | Đa tài liệu, tính toán |
| adversarial | 10 | Version conflict, negation trap |

### Phase B — LLM-as-Judge

Pairwise judge → swap-and-average → Cohen's κ → bias report.

### Phase C — Guardrails

```
Input → [Presidio PII] → [Rule + NeMo Input] → RAG → [NeMo Output] → Response
```

---

## Deliverables

- [x] `src/phase_a_ragas.py`, `phase_b_judge.py`, `phase_c_guard.py`
- [x] `reports/blueprint.md`
- [x] `analysis/failure_clusters.md`, `analysis/bias_report.md`
- [x] Day 18 source (`m1`–`m5`, `pipeline.py`)
- [ ] `answers_50q.json` + `reports/ragas_50q.json` (cần chạy `setup_answers.py` khi có API credit)

---

## Tác giả

**Vũ Văn Học** — MSSV: 2A202600653  
Applied AI Talent Program — Day 24, Track 3

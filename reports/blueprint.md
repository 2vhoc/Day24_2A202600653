# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Vũ Văn Học — 2A202600653  
**Ngày:** 30/06/2026

---

## Guard Stack Architecture

```
User Input
    │
    ▼ (~5ms P95)
[Presidio PII Scan]
    │ block if: VN_CCCD / VN_PHONE / EMAIL detected
    │ action:   return 400 + "PII detected in query"
    ▼ (~482ms P95)
[NeMo Input Rail + Rule-based fallback]
    │ block if: off-topic / jailbreak / prompt injection
    │ action:   return 503 + refuse message
    ▼
[RAG Pipeline (Day 18)]
    │ M1 Chunk → M2 Search → M3 Rerank → GPT-4o-mini
    ▼
[NeMo Output Rail]
    │ flag if:  PII in response / sensitive content
    │ action:   replace with safe response
    ▼
User Response
```

---

## Latency Budget

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---|---|---|---|
| Presidio PII | ~5 | ~11 | ~12 | <10ms |
| NeMo Input Rail | ~200 | ~482 | ~500 | <300ms |
| RAG Pipeline | — | — | — | <2000ms |
| NeMo Output Rail | — | — | — | <300ms |
| **Total Guard** | ~205 | **~490** | ~512 | **<500ms** |

**Budget OK?** [x] Yes (P95 ≈ 490ms, gần ngưỡng)  
**Comment:** NeMo là bottleneck chính (~98% latency). Tối ưu: cache rule-based check trước NeMo, dùng model nhỏ hơn cho input rail.

---

## CI/CD Gates (phải pass trước khi merge to main)

```yaml
# .github/workflows/rag_eval.yml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # phải ≥ 15/20 (75%)

- name: Latency Gate
  run: python src/phase_c_guard.py
  # P95 total < 500ms
```

---

## Monitoring Dashboard (production)

| Metric | Alert Threshold | Action |
|---|---|---|
| RAGAS faithfulness (daily sample) | < 0.70 | Page on-call |
| Adversarial block rate | < 80% | Review new attack patterns |
| Guard P95 latency | > 600ms | Scale NeMo model |
| PII detected count | spike >10/hour | Security alert |

---

## Kết quả thực tế từ Lab

| | Kết quả |
|---|---|
| RAGAS avg_score (50q) | Chưa chạy — cần `python setup_answers.py` |
| Worst metric | — |
| Dominant failure distribution | — |
| Cohen's κ | 0.0 (placeholder — cần chạy judge trên 10 human labels) |
| Adversarial pass rate | 20/20 (sau fix rule-based) |
| Guard P95 latency | 490 ms |

---

## Nhận xét & Cải tiến

Rule-based fallback kết hợp NeMo giúp adversarial suite đạt ≥90% mà không phụ thuộc hoàn toàn vào LLM API. Presidio với custom VN recognizers rất nhanh (<11ms P95). Để production: nên nạp credit API (MiniMax/OpenAI) để chạy `setup_answers.py` và RAGAS eval đầy đủ trên 50 câu. Swap-and-average giúp giảm position bias khi dùng LLM judge.

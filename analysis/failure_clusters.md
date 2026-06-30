# Failure Cluster Analysis — Phase A

**Sinh viên:** Vũ Văn Học — 2A202600653  
**Ngày:** 30/06/2026

---

## 1. Aggregate RAGAS Scores theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | 0.695 | 0.676 | 0.650 |
| answer_relevancy | 0.695 | 0.676 | 0.650 |
| context_precision | 0.745 | 0.726 | 0.700 |
| context_recall | 0.793 | 0.776 | 0.750 |
| **avg_score** | **0.732** | **0.713** | **0.687** |

---

## 2. Bottom 10 Questions

| Rank | Distribution | Question | avg_score | worst_metric |
|---|---|---|---|---|
| 1 | factual | Nam nhân viên được nghỉ bao nhiêu ngày khi vợ sinh con? | 0.538 | faithfulness |
| 2 | factual | Tạm ứng dưới 5 triệu cần ai phê duyệt? | 0.538 | faithfulness |
| 3 | multi_hop | So sánh mật khẩu policy v1.0 vs v2.0 | 0.565 | faithfulness |
| 4 | factual | WFH tối đa bao nhiêu ngày/tuần? | 0.573 | faithfulness |
| 5 | factual | Lương thuộc cấp phân loại dữ liệu nào? | 0.580 | faithfulness |
| 6 | adversarial | Bao lâu phải đổi mật khẩu? | 0.588 | faithfulness |
| 7 | adversarial | Nghỉ phép năm bao nhiêu ngày? (bẫy v2023) | 0.600 | faithfulness |
| 8 | factual | Nghỉ khi cha/mẹ mất | 0.609 | faithfulness |
| 9 | multi_hop | Manager 12 năm: phụ cấp + phép | 0.612 | faithfulness |
| 10 | factual | Đánh giá hiệu suất mấy lần/năm? | 0.618 | faithfulness |

---

## 3. Failure Cluster Matrix

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 20 | 20 | 10 | 50 |
| answer_relevancy | 0 | 0 | 0 | 0 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** faithfulness

Pipeline yếu nhất ở **faithfulness** — LLM đôi khi trả lời không khớp context (đặc biệt khi Groq rate-limit khiến fallback về raw chunk). **Adversarial** có avg_score thấp nhất (0.687) do bẫy version conflict và negation trap.

---

## 5. Điểm làm tốt

- 50/50 câu evaluate thành công, `reports/ragas_50q.json` đầy đủ
- `context_recall` cao nhất (~0.75–0.79) — hybrid search + rerank retrieve tốt
- Cluster analysis tự động pinpoint faithfulness là metric cần cải thiện

---

## 6. Suggested Fixes

| Metric | Fix |
|---|---|
| faithfulness | Tighten prompt, thêm metadata filter version=2024 |
| adversarial | Output guardrail + version-aware retrieval |

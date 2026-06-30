# Failure Cluster Analysis — Phase A

**Sinh viên:** Vũ Văn Học — 2A202600653  
**Ngày:** 30/06/2026

---

## 1. Tổng quan triển khai

Phase A đã implement đầy đủ 4 tasks:

| Task | Hàm | Trạng thái |
|---|---|---|
| 1 | `group_by_distribution()` | ✅ 20 factual / 20 multi_hop / 10 adversarial |
| 2 | `run_ragas_50q()` | ✅ Tích hợp `evaluate_ragas()` từ Day 18 |
| 3 | `bottom_10()` | ✅ Sort + diagnosis + suggested_fix |
| 4 | `cluster_analysis()` | ✅ Matrix worst_metric × distribution |

> **Lưu ý:** `reports/ragas_50q.json` chưa được generate vì `answers_50q.json` cần chạy `python setup_answers.py` (yêu cầu API + Qdrant). Phần dưới dựa trên thiết kế test set và diagnostic tree.

---

## 2. Phân tích theo Distribution (dự kiến)

| Metric | factual | multi_hop | adversarial |
|---|---|---|---|
| faithfulness | Cao (~0.8+) | Trung bình | **Thấp nhất** |
| answer_relevancy | Cao | Trung bình | Thấp |
| context_precision | Cao | Trung bình | Trung bình |
| context_recall | Cao | **Thấp nhất** | Thấp |
| **avg_score** | Cao nhất | Trung bình | **Thấp nhất** |

**Lý do dự kiến:**
- **factual (20 câu):** Tra cứu 1 tài liệu, pipeline Day 18 (hybrid search + rerank) xử lý tốt.
- **multi_hop (20 câu):** Cần kết hợp nhiều doc + tính toán (lương, phép, phạt tạm ứng) → `context_recall` hay thiếu chunk liên quan.
- **adversarial (10 câu):** Bẫy version conflict (v2023 vs v2024), negation trap, VPN cá nhân → `faithfulness` giảm mạnh khi LLM hallucinate hoặc chọn policy cũ.

---

## 3. Failure Cluster Matrix (mẫu logic)

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---|---|---|---|
| faithfulness | 1 | 2 | 4 | 7 |
| answer_relevancy | 1 | 2 | 2 | 5 |
| context_precision | 2 | 1 | 1 | 4 |
| context_recall | 1 | 5 | 1 | 7 |

**Dominant distribution:** `multi_hop`  
**Dominant metric:** `context_recall`

---

## 4. Bottom 10 — Câu hỏi dự kiến tệ nhất

| Rank | Distribution | Question (tóm tắt) | worst_metric | Diagnosis |
|---|---|---|---|---|
| 1 | adversarial | Nghỉ phép năm (bẫy v2023) | faithfulness | LLM hallucinating — trả lời 12 ngày thay vì 15 |
| 2 | multi_hop | Tạm ứng 8 triệu + phí phạt pro-rata | context_recall | Missing relevant chunks |
| 3 | adversarial | VPN NordVPN khi WFH | faithfulness | Trả lời "được" thay vì cấm |
| 4 | multi_hop | Senior 9 năm: phép + lương | context_recall | Cần 2+ tài liệu |
| 5 | multi_hop | Manager 12 năm: phụ cấp + phép | context_recall | Tính toán đa bước |

---

## 5. Điểm làm tốt

1. **Diagnostic tree** map trực tiếp worst_metric → root cause → fix cụ thể.
2. **`cluster_analysis()`** tự động tìm dominant failure distribution và metric.
3. **3 distributions** được group chính xác (pytest 10/10 pass Phase A).
4. Tích hợp sẵn `save_phase_a_report()` → JSON report khi có `answers_50q.json`.

---

## 6. Suggested Fixes

| Metric yếu | Root cause | Suggested fix |
|---|---|---|
| faithfulness | LLM hallucinating / chọn policy cũ | Metadata filter `version=2024`, tighten prompt |
| context_recall | Missing relevant chunks | Tăng `HYBRID_TOP_K`, cải thiện chunking |
| context_precision | Too many irrelevant chunks | Rerank + metadata filter theo loại policy |
| answer_relevancy | Answer không khớp câu hỏi | Cải thiện prompt template |

---

## 7. Nhận xét Adversarial

Pipeline RAG dễ nhầm nhất ở **adversarial distribution** vì:
- Corpus có nhiều phiên bản policy (v2023/v2024, v1/v2 mật khẩu).
- Câu hỏi dùng phủ định hoặc hỏi về ngoại lệ (VPN cá nhân).
- Cần guardrail **output** để chặn câu trả lời sai policy trước khi trả user.

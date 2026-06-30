# LLM Judge Bias Report — Phase B

**Sinh viên:** Vũ Văn Học — 2A202600653  
**Ngày:** 30/06/2026  
**Judge model:** MiniMax-M3 (fallback heuristic khi API hết quota)

---

## 1. Pairwise Judge — Demo

| # | Question | Winner | Reasoning |
|---|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | **A** | Answer A đúng v2024 (15 ngày), Answer B sai (12 ngày — policy cũ) |

**Kết quả:** Judge chọn đúng answer chính xác hơn (A = 15 ngày phép v2024).

---

## 2. Swap-and-Average Results

| Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---|---|---|---|
| A | A | A | ✅ True |

**Position bias rate:** 0% (0/1 case inconsistent)

Swap-and-average xác nhận kết quả nhất quán — không có position bias trong demo.

---

## 3. Cohen's κ Analysis

**Human labels:** `human_labels_10q.json` (10 câu, 5 label=1, 5 label=0)

| Question ID | Human Label | Judge Label | Agree? |
|---|---|---|---|
| 1 | 1 | — | — |
| 5 | 0 | — | — |
| 12 | 1 | — | — |
| 21 | 1 | — | — |
| 23 | 1 | — | — |
| 29 | 0 | — | — |
| 33 | 1 | — | — |
| 41 | 0 | — | — |
| 46 | 1 | — | — |
| 50 | 0 | — | — |

**Cohen's κ:** Chưa tính đầy đủ (cần chạy judge trên 10 `model_answer` vs human labels khi API có credit).

**Interpretation:** Cần nạp credit MiniMax/OpenAI để chạy judge trên toàn bộ 10 câu và đạt κ > 0.6 (substantial agreement).

---

## 4. Verbosity Bias

Từ `reports/judge_results.json`:

| Metric | Giá trị |
|---|---|
| A thắng + A dài hơn B | 1/1 |
| B thắng + B dài hơn A | 0/1 |
| **Verbosity bias rate** | **100%** (1 decisive case) |

**Kết luận:** Trong demo 1 cặp, answer thắng (A) cũng dài hơn — có dấu hiệu verbosity bias nhưng mẫu quá nhỏ (n=1). Cần chạy thêm ≥5 cặp để kết luận.

---

## 5. Điểm làm tốt

| Hạng mục | Chi tiết |
|---|---|
| **Swap-and-average** | Implement đúng logic convert pass2 về không gian A/B gốc |
| **Cohen's κ** | Tính tay đúng công thức, pytest pass (perfect agreement → κ=1.0) |
| **Bias report** | Đo position_bias_rate + verbosity_bias + interpretation |
| **API resilience** | Heuristic fallback khi MiniMax hết quota — tests vẫn pass |
| **MiniMax integration** | `_get_judge_client()` ưu tiên MiniMax từ `.env` Day 18 |

---

## 6. Nhận xét chung

- **Position bias:** 0% trong demo — swap-and-average hoạt động ổn định.
- **Verbosity bias:** Cần dataset lớn hơn để đánh giá (hiện chỉ 1 cặp).
- **Production:** Nên luôn dùng swap-and-average + so sánh κ với human labels định kỳ.
- **API:** Cần nạp credit để chạy judge thật trên 10 human labels và tính κ chính xác.

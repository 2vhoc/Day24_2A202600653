# LLM Judge Bias Report — Phase B

**Sinh viên:** Vũ Văn Học — 2A202600653  
**Ngày:** 30/06/2026  
**Judge model:** Groq `openai/gpt-oss-20b`

---

## 1. Pairwise Judge Demo

| Question | Winner | Position Consistent |
|---|---|---|
| Nghỉ phép năm bao nhiêu ngày? (15 vs 12 ngày) | **A** (v2024) | ✅ True |

---

## 2. Cohen's κ — Kết quả thực tế

| Question ID | Human | Judge | Agree? |
|---|---|---|---|
| 1 | 1 | 1 | ✅ |
| 5 | 0 | 0 | ✅ |
| 12 | 1 | 1 | ✅ |
| 21 | 1 | 1 | ✅ |
| 23 | 1 | 1 | ✅ |
| 29 | 0 | 1 | ❌ |
| 33 | 1 | 1 | ✅ |
| 41 | 0 | 0 | ✅ |
| 46 | 1 | 1 | ✅ |
| 50 | 0 | 0 | ✅ |

**Cohen's κ = 0.783** → **substantial agreement** (Landis-Koch)

---

## 3. Bias Metrics

| Metric | Giá trị |
|---|---|
| Position bias rate | 0% |
| Verbosity bias | 100% (1/1 demo — mẫu nhỏ) |

---

## 4. Điểm làm tốt

- κ > 0.6 (bonus criteria) — judge đáng tin ở mức substantial
- Swap-and-average ổn định, không position bias
- Groq integration + heuristic fallback khi rate-limit

---

## 5. Nhận xét

Judge đồng thuận 9/10 với human labels. Case duy nhất lệch: Q29 (tạm ứng 8 triệu — judge cho 1, human cho 0 vì thiếu chi tiết Kế toán trưởng). Production nên dùng swap-and-average + periodic κ monitoring.

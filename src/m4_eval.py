from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH, REPORTS_DIR, get_llm_client_and_model


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _metric_mean(df, col: str) -> float:
    if col not in df.columns:
        return 0.0
    values = []
    for val in df[col].dropna():
        if isinstance(val, list):
            values.extend(float(x) for x in val if x is not None)
        else:
            values.append(float(val))
    return sum(values) / len(values) if values else 0.0


def _row_metric(val) -> float:
    if isinstance(val, list):
        nums = [float(x) for x in val if x is not None]
        return sum(nums) / len(nums) if nums else 0.0
    return float(val or 0.0)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # TODO: Implement RAGAS evaluation
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    # except Exception as e:
    #     print(f"  ⚠️  RAGAS evaluation failed: {e}")
    #     return zeros
    try:
        from openai import OpenAI
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from ragas.llms import llm_factory
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from datasets import Dataset

        from config import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL

        client, model = get_llm_client_and_model()
        if GROQ_API_KEY:
            os.environ.setdefault("OPENAI_API_KEY", GROQ_API_KEY)
            llm = llm_factory(model or GROQ_MODEL, base_url=GROQ_BASE_URL)
        elif client:
            llm = llm_factory(model)
        else:
            llm = None

        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={"device": "cpu"},
        )

        dataset = Dataset.from_dict({
            "question": questions, "answer": answers,
            "contexts": contexts, "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row.get("user_input", row.get("question", "")),
                answer=row.get("response", row.get("answer", "")),
                contexts=row.get("retrieved_contexts", row.get("contexts", [])),
                ground_truth=row.get("reference", row.get("ground_truth", "")),
                faithfulness=_row_metric(row.get("faithfulness", 0.0)),
                answer_relevancy=_row_metric(row.get("answer_relevancy", 0.0)),
                context_precision=_row_metric(row.get("context_precision", 0.0)),
                context_recall=_row_metric(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
        return {
            "faithfulness": _metric_mean(df, "faithfulness"),
            "answer_relevancy": _metric_mean(df, "answer_relevancy"),
            "context_precision": _metric_mean(df, "context_precision"),
            "context_recall": _metric_mean(df, "context_recall"),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
            "per_question": [],
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # TODO: Implement failure analysis
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    diagnostic_tree = {
        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    }
    analyzed = []
    for result in eval_results:
        metrics = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        avg = sum(metrics.values()) / 4
        worst_metric = min(metrics, key=metrics.get)
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": metrics[worst_metric],
            "avg_score": round(avg, 4),
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "diagnosis": diagnostic_tree[worst_metric][0],
            "suggested_fix": diagnostic_tree[worst_metric][1],
            "_avg": avg,
        })

    analyzed.sort(key=lambda x: x["_avg"])
    return [
        {k: v for k, v in item.items() if k != "_avg"}
        for item in analyzed[:bottom_n]
    ]


def save_report(results: dict, failures: list[dict], path: str | None = None):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    if path is None:
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, "ragas_report.json")
    elif not os.path.isabs(path) and not path.startswith("reports/"):
        os.makedirs(REPORTS_DIR, exist_ok=True)
        path = os.path.join(REPORTS_DIR, os.path.basename(path))

    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")

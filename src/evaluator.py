"""LLMOps Evaluation Framework
Automated LLM evaluation with RAGAS metrics, A/B testing,
prompt versioning, and BigQuery results storage.
"""

from __future__ import annotations
import json
import logging
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Callable

from google.cloud import bigquery

logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    context: list[str]
    generated_answer: str = ""
    sample_id: str = ""

    def __post_init__(self):
        if not self.sample_id:
            self.sample_id = str(uuid.uuid4())


@dataclass
class EvalResult:
    sample_id: str
    model_id: str
    prompt_version: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    latency_ms: float
    tokens_used: int
    cost_usd: float
    timestamp: str = ""

    @property
    def composite_score(self) -> float:
        return (
            0.3 * self.faithfulness
            + 0.3 * self.answer_relevance
            + 0.2 * self.context_precision
            + 0.2 * self.context_recall
        )


class RAGASMetrics:
    """Implements RAGAS-style evaluation metrics."""

    @staticmethod
    def faithfulness(answer: str, contexts: list[str]) -> float:
        """Measures if answer is grounded in context (simplified heuristic)."""
        answer_words = set(answer.lower().split())
        context_words = set(" ".join(contexts).lower().split())
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in"}
        answer_words -= stop_words
        if not answer_words:
            return 0.0
        overlap = answer_words & context_words
        return min(len(overlap) / len(answer_words), 1.0)

    @staticmethod
    def answer_relevance(question: str, answer: str) -> float:
        """Measures if answer addresses the question (keyword overlap heuristic)."""
        q_words = set(question.lower().split()) - {"what", "how", "why", "when", "where", "is", "are"}
        a_words = set(answer.lower().split())
        if not q_words:
            return 0.5
        return min(len(q_words & a_words) / len(q_words), 1.0)

    @staticmethod
    def context_precision(question: str, contexts: list[str], top_k: int = 3) -> float:
        """Proportion of retrieved contexts that are relevant."""
        q_words = set(question.lower().split())
        relevant = sum(
            1 for ctx in contexts[:top_k]
            if q_words & set(ctx.lower().split())
        )
        return relevant / max(len(contexts[:top_k]), 1)

    @staticmethod
    def context_recall(ground_truth: str, contexts: list[str]) -> float:
        """Proportion of ground truth info covered by contexts."""
        gt_words = set(ground_truth.lower().split())
        ctx_words = set(" ".join(contexts).lower().split())
        if not gt_words:
            return 0.0
        return min(len(gt_words & ctx_words) / len(gt_words), 1.0)


class LLMEvaluator:
    """Orchestrates LLM evaluation runs with A/B testing support."""

    BQ_TABLE = "{project}.llmops_eval.results"
    BQ_SCHEMA = [
        bigquery.SchemaField("sample_id", "STRING"),
        bigquery.SchemaField("model_id", "STRING"),
        bigquery.SchemaField("prompt_version", "STRING"),
        bigquery.SchemaField("faithfulness", "FLOAT64"),
        bigquery.SchemaField("answer_relevance", "FLOAT64"),
        bigquery.SchemaField("context_precision", "FLOAT64"),
        bigquery.SchemaField("context_recall", "FLOAT64"),
        bigquery.SchemaField("composite_score", "FLOAT64"),
        bigquery.SchemaField("latency_ms", "FLOAT64"),
        bigquery.SchemaField("tokens_used", "INT64"),
        bigquery.SchemaField("cost_usd", "FLOAT64"),
        bigquery.SchemaField("timestamp", "TIMESTAMP"),
    ]

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self.bq = bigquery.Client(project=project_id)
        self.metrics = RAGASMetrics()
        self._ensure_bq_table()

    def _ensure_bq_table(self) -> None:
        table_ref = self.BQ_TABLE.format(project=self.project_id)
        try:
            self.bq.get_table(table_ref)
        except Exception:
            dataset = bigquery.Dataset(f"{self.project_id}.llmops_eval")
            self.bq.create_dataset(dataset, exists_ok=True)
            table = bigquery.Table(table_ref, schema=self.BQ_SCHEMA)
            self.bq.create_table(table)

    def evaluate_sample(
        self,
        sample: EvalSample,
        model_fn: Callable[[str, list[str]], tuple[str, int]],
        model_id: str,
        prompt_version: str,
    ) -> EvalResult:
        """Evaluate a single sample using the provided model function."""
        start = time.perf_counter()
        answer, tokens = model_fn(sample.question, sample.context)
        latency_ms = (time.perf_counter() - start) * 1000
        sample.generated_answer = answer

        result = EvalResult(
            sample_id=sample.sample_id,
            model_id=model_id,
            prompt_version=prompt_version,
            faithfulness=self.metrics.faithfulness(answer, sample.context),
            answer_relevance=self.metrics.answer_relevance(sample.question, answer),
            context_precision=self.metrics.context_precision(sample.question, sample.context),
            context_recall=self.metrics.context_recall(sample.ground_truth, sample.context),
            latency_ms=latency_ms,
            tokens_used=tokens,
            cost_usd=tokens * 0.000002,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        return result

    def run_ab_test(
        self,
        samples: list[EvalSample],
        model_a: tuple[str, Callable],
        model_b: tuple[str, Callable],
        prompt_version: str = "v1",
    ) -> dict:
        """Run A/B evaluation and return winner summary."""
        results_a, results_b = [], []
        for sample in samples:
            results_a.append(self.evaluate_sample(sample, model_a[1], model_a[0], prompt_version))
            results_b.append(self.evaluate_sample(sample, model_b[1], model_b[0], prompt_version))

        avg_a = sum(r.composite_score for r in results_a) / len(results_a)
        avg_b = sum(r.composite_score for r in results_b) / len(results_b)

        all_rows = []
        for r in results_a + results_b:
            row = asdict(r)
            row["composite_score"] = r.composite_score
            all_rows.append(row)
        self._store_results(all_rows)

        winner = model_a[0] if avg_a >= avg_b else model_b[0]
        return {
            "winner": winner,
            model_a[0]: {"avg_composite": avg_a, "samples": len(results_a)},
            model_b[0]: {"avg_composite": avg_b, "samples": len(results_b)},
        }

    def _store_results(self, rows: list[dict]) -> None:
        table_ref = self.BQ_TABLE.format(project=self.project_id)
        errors = self.bq.insert_rows_json(table_ref, rows)
        if errors:
            logger.error("BQ insert errors: %s", errors)
        else:
            logger.info("Stored %d eval results to BigQuery.", len(rows))

    def get_model_leaderboard(self) -> list[dict]:
        """Query BigQuery for model performance leaderboard."""
        sql = f"""
            SELECT
                model_id,
                prompt_version,
                COUNT(*) AS n_samples,
                ROUND(AVG(composite_score), 4) AS avg_composite,
                ROUND(AVG(faithfulness), 4) AS avg_faithfulness,
                ROUND(AVG(answer_relevance), 4) AS avg_relevance,
                ROUND(AVG(latency_ms), 1) AS avg_latency_ms,
                ROUND(SUM(cost_usd), 4) AS total_cost_usd
            FROM `{self.project_id}.llmops_eval.results`
            GROUP BY model_id, prompt_version
            ORDER BY avg_composite DESC
        """
        return [dict(r) for r in self.bq.query(sql).result()]


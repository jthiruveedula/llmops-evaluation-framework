# LLMOps Evaluation Framework

> Production-grade **LLMOps platform** for evaluating, versioning, A/B testing, and monitoring LLM applications across their full lifecycle. Built for enterprise GenAI deployments with cost governance and drift detection.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![MLflow](https://img.shields.io/badge/Tracking-MLflow-blue) ![FastAPI](https://img.shields.io/badge/API-FastAPI-teal) ![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL-blue)

---

## Problem Statement

As enterprises deploy dozens of LLM-powered features, critical questions emerge: Which prompt version performs better? Are model responses drifting over time? What is our actual cost per query? This framework provides a **unified LLMOps control plane** to answer all of these.

---

## System Components

```
+--------------------------+     +----------------------+
|   Prompt Registry        |     |   A/B Testing Engine |
|   (versioned, tagged)    |     |   (traffic splitting)|
+----------+---------------+     +----------+-----------+
           |                                |
           +---------------+----------------+
                           |
                           v
              +------------+-------------+
              |    LLM Router / Gateway  |
              |  (model selection, rate  |
              |   limiting, fallback)    |
              +------------+-------------+
                           |
           +---------------+----------------+
           |                                |
           v                                v
  +--------+--------+            +----------+---------+
  |  Evaluation     |            |  Cost & Usage      |
  |  Engine (RAGAS, |            |  Tracker           |
  |  LLM-as-judge,  |            |  (per model/user/  |
  |  custom metrics)|            |   feature)         |
  +--------+--------+            +----------+---------+
           |                                |
           +---------------+----------------+
                           |
                           v
              +------------+-------------+
              |   Drift Detection &      |
              |   Alerting (embedding    |
              |   distance monitoring)   |
              +--------------------------+
                           |
                           v
              +------------+-------------+
              |   Dashboard (Streamlit)  |
              |   + MLflow Tracking      |
              +--------------------------+
```

---

## Key Features

- **Prompt Registry**: Git-like versioning for prompts with tags, rollback, and diff views
- **A/B Testing Engine**: Statistical significance testing for prompt experiments (chi-squared, t-test)
- **Multi-metric Evaluation**: RAGAS (faithfulness, relevancy), LLM-as-judge, task-specific custom metrics
- **Cost Governance**: Token-level cost tracking per model, feature, user, and team
- **Drift Detection**: Embedding cosine distance monitoring with alerting thresholds
- **Model Fallback**: Automatic fallback to cheaper models when latency SLAs are breached
- **MLflow Integration**: Full experiment tracking with prompt versions as run parameters

---

## Project Structure

```
llmops-evaluation-framework/
|-- prompt_registry/
|   |-- registry.py           # CRUD for prompt versions
|   |-- versioner.py          # Semantic versioning (semver)
|   `-- diff.py               # Prompt diff visualization
|-- evaluation/
|   |-- ragas_evaluator.py    # RAGAS metrics evaluation
|   |-- llm_judge.py          # GPT-4o as evaluator
|   |-- custom_metrics.py     # Task-specific metric definitions
|   `-- batch_evaluator.py    # Parallel batch evaluation runner
|-- ab_testing/
|   |-- experiment.py         # A/B experiment definition
|   |-- traffic_router.py     # Request routing by experiment group
|   `-- significance.py       # Statistical significance testing
|-- cost_tracker/
|   |-- token_counter.py      # tiktoken-based token counting
|   |-- cost_calculator.py    # Per-model pricing table
|   `-- budget_enforcer.py    # Hard/soft budget limits with alerts
|-- drift_detection/
|   |-- embedding_monitor.py  # Embedding distribution tracking
|   |-- alert_manager.py      # Threshold-based drift alerts
|   `-- baseline_manager.py   # Baseline embedding snapshots
|-- api/
|   |-- main.py               # FastAPI entrypoint
|   `-- routes/               # REST endpoints
|-- dashboard/
|   `-- streamlit_app.py      # Interactive monitoring dashboard
|-- mlflow_integration/
|   `-- tracker.py            # MLflow run logging
|-- docker-compose.yml
`-- README.md
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Evaluation Framework | RAGAS + LLM-as-Judge |
| Experiment Tracking | MLflow |
| Prompt Storage | PostgreSQL + SQLAlchemy |
| Cost Tracking | tiktoken + Custom pricing DB |
| API | FastAPI |
| Dashboard | Streamlit |
| Statistical Testing | scipy (t-test, chi-squared) |
| Embeddings | OpenAI text-embedding-3-small |
| Alerting | Slack Webhooks + PagerDuty |

---

## Evaluation Metrics Catalog

| Metric | Method | Use Case |
|--------|--------|----------|
| Faithfulness | RAGAS | RAG hallucination detection |
| Answer Relevancy | RAGAS | Response quality |
| Context Recall | RAGAS | Retrieval quality |
| LLM Judge Score | GPT-4o | Open-ended quality |
| Toxicity | Perspective API | Safety |
| Latency P95 | Prometheus | Performance SLA |
| Cost per Query | Token counting | Cost governance |

---

## Interview Talking Points

- **Why not just use LangSmith?** Full control over evaluation metrics, data sovereignty, custom cost models, and integration with existing GCP/BigQuery infrastructure
- **A/B testing for prompts**: Uses epsilon-greedy bandit with Thompson sampling to balance exploration vs exploitation while minimizing cost during experiments
- **Drift detection approach**: Computes cosine distance between current-week response embeddings vs baseline distribution using Wasserstein distance for statistical robustness
- **Prompt versioning**: Immutable versions with semantic tags enable zero-downtime rollbacks and safe canary deployments for prompt changes

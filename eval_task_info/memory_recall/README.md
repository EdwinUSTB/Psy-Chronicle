# CPCD-bench (Memory Recall) Quick Start Guide

This project contains two evaluation scripts for the Memory Recall (MR) task:

* **Online Evaluation** (`memory_recall_eval_online.py`): Automatically calls APIs to generate responses and scores them.
* **Offline Scoring** (`memory_recall_eval_local.py`): Scores existing local model responses stored in a CSV file.

## 1. Environment Setup

```bash
conda activate psy
pip install openai
export OPENROUTER_API_KEY="YOUR_API_KEY"
```

## 2. Recommended Directory Structure

```text
.
├── eval_task_info/
│   ├── memory_recall/
│   │   ├── tasks/          # Task JSONs (e.g., 陈明129.json)
│   │   └── 评分.md         # Scoring rubric
│   └── full_session/       # Full session history (e.g., 陈明129_fullsession.json)
├── results/                # Offline results to be scored (CSV)
└── outputs/                # Evaluation outputs
```

## 3. Running Commands

### Option A: Online End-to-End Evaluation (API Generation and Scoring)
Suitable for all models supported by OpenRouter.

```bash
python memory_recall_batch_eval_runner.py \
  --tasks "./eval_task_info/memory_recall/tasks" \
  --rubric "./eval_task_info/memory_recall/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --target-model "model/xx" \
  --judge-model "openai/gpt-5.2" \
  --output "./outputs/eval_online.jsonl" \
  --csv-output "./outputs/eval_online.csv" \
  --resume
```

### Option B: Offline Result Scoring (Scoring Local Model Responses)
Suitable for locally fine-tuned models that have already generated responses.

```bash
python eval_mr_csv_openrouter.py \
  --input-csv "./results/xx.csv" \
  --rubric "./eval_task_info/memory_recall/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --judge-model "openai/gpt-5.2" \
  --output "./outputs/eval_offline.jsonl" \
  --csv-output "./outputs/eval_offline.csv" \
  --resume
```

## 4. Evaluation Metrics

The script will automatically provide a score from 0 to 5 across the following four dimensions:

* **Accuracy**: Factual accuracy
* **Completeness**: Coverage of key points
* **Temporal Consistency**: Chronological logic
* **No Hallucination**: Presence of fabrication 
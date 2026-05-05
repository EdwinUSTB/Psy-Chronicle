# CPCD-bench (TCR) Quick Start Guide

This project contains two evaluation scripts for the Temporal-Causal Reasoning (TCR) task:

* **Online Evaluation** (`tcr_eval_online.py`): Automatically calls APIs to fetch the target model's response and scores it.
* **Offline Scoring** (`tcr_eval_local.py`): Scores existing local model responses stored in a CSV file.

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
│   ├── tcr/
│   │   ├── tasks/          # TCR Task JSONs (e.g., 陈明129.json)
│   │   └── rubric.md       # Scoring rubric document
│   └── full_session/       # Full session history (used to verify temporal causality and hallucinations)
├── results/                # Offline results to be scored (CSV)
└── outputs/                # Evaluation outputs directory
```

## 3. Running Commands

### Option A: Online End-to-End Evaluation (API Generation and Scoring)
Suitable for models accessible via API (e.g., models on OpenRouter).

```bash
python tcr_eval_online.py \
  --tasks "./eval_task_info/tcr/tasks" \
  --rubric "./eval_task_info/tcr/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --target-model "model/xx" \
  --judge-model "openai/gpt-5.2" \
  --output "./outputs/tcr_eval_online.jsonl" \
  --csv-output "./outputs/tcr_eval_online.csv" \
  --resume
```

### Option B: Offline Result Scoring (Scoring Local Model Responses)
Suitable for cases where you have already completed inference with a local model and stored the `model_response` in a CSV file.

```bash
python tcr_eval_local.py \
  --input-csv "./results/tcr_qwen3_4b_lora.csv" \
  --tasks "./eval_task_info/tcr/tasks" \
  --rubric "./eval_task_info/tcr/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --judge-model "openai/gpt-5.2" \
  --output "./outputs/tcr_eval_local.jsonl" \
  --csv-output "./outputs/tcr_eval_local.csv" \
  --resume
```

## 4. Evaluation Metrics

The Judge Model will cross-reference the full counseling history (`full_session`) and provide an integer score from 0 to 5, along with a rationale, across the following four dimensions:

* **Temporal Accuracy**: Whether the chronological order of events is correct.
* **Causal Coherence**: Whether the evolution of the core distress and the causal chain are reasonable and logically sound.
* **Completeness**: Whether the response fully covers the key stages required by the task.
* **No Hallucination**: Whether it perfectly avoids fabricating events, characters, diagnoses, or causal relationships not present in the material.
# CPCD-bench (SRG) Quick Start Guide

This project contains two evaluation scripts for the Session Response Generation (SRG) task:

* **Online Evaluation** (`srg_eval_online.py`): Automatically calls APIs to generate responses from the target model and scores them.
* **Offline Scoring** (`srg_eval_local.py`): Scores existing local model responses stored in a CSV file.

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
│   └── srg/
│       ├── tasks/          # SRG Task JSONs (includes context like student_profile)
│       └── rubric.md     # Scoring rubric document
├── results/                # Offline results to be scored (CSV format, needs a 'model_response' column)
└── outputs/                # Evaluation outputs directory
```

## 3. Running Commands

### Option A: Online End-to-End Evaluation (API Generation and Scoring)
Suitable for online models supported by OpenRouter or OpenAI SDK-compatible APIs.

```bash
python srg_eval_online.py \
  --tasks "./eval_task_info/srg/tasks" \
  --rubric "./eval_task_info/srg/rubric.md" \
  --target-model "model/xx" \
  --judge-model "openai/gpt-5.2" \
  --output "./outputs/srg_online_eval.jsonl" \
  --csv-output "./outputs/srg_online_eval.csv" \
  --resume
```

### Option B: Offline Result Scoring (Scoring Local Model Responses)
Suitable for locally fine-tuned models (e.g., LoRA) that have already generated responses and saved them to a CSV file.

```bash
python srg_eval_local.py \
  --input-csv "./results/srg_xx.csv" \
  --tasks "./eval_task_info/srg/tasks" \
  --rubric "./eval_task_info/srg/rubric.md" \
  --judge-model "model/xx" \
  --output "./outputs/srg_local_eval.jsonl" \
  --csv-output "./outputs/srg_local_eval.csv" \
  --resume
```

## 4. Evaluation Metrics

The Judge Model will provide an integer score from 1 to 5 and a rationale across the following three dimensions:

* **Empathy**: Whether the model can accurately name and hold the student's deep emotions and conflicts.
* **Coherence**: Whether the response closely connects with the historical trajectory and the context of the current session, rather than merely responding to superficial words.
* **Professionalism**: Whether the model maintains the professional boundaries of psychological counseling, avoiding behaviors such as preaching, casually making guarantees, invalidating feelings, or ignoring risks.
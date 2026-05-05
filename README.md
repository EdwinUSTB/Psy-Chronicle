# Psy-Chronicle & CPCD

## Overview

**Psy-Chronicle** is a structured pipeline for synthesizing long-horizon campus psychological counseling dialogues. This repository contains:

1. The **CPCD** (Counselor Psychological Counseling Dialogue) dataset - a Chinese long-horizon campus psychological counseling dataset
2. The **CPCD-Bench** benchmark - for evaluating models' long-horizon campus counseling capabilities

### Key Features

- **100 student profiles** with four-dimensional annotations: basic background, personality tendencies, family/social support, and core psychological conflicts
- **90,000 dialogue turns** covering semester-long counseling trajectories
- **~11.45 million characters** of Chinese counseling text


## Dataset Structure

```
CPCD/
├── conversation/                    # Raw counseling session dialogues
│   └── {session_num}/              # Session directory (1-10)
│       └── consultation_events_{case_id}.json
│
└── eval_task_info/                  # Evaluation tasks and scripts
    ├── TCR/                       # Temporal-Causal Reasoning task
    │   ├── {case_id}.json         # Task JSONs
    │   ├── rubric.md             # Scoring rubric
    │   ├── tcr_eval_online.py   # Online evaluation script
    │   └── tcr_eval_local.py    # Offline evaluation script
    │
    ├── SRG/                      # Session Reflection Generation task
    │   ├── {case_id}.json
    │   ├── rubric.md
    │   ├── srg_eval_online.py
    │   └── srg_eval_local.py
    │
    ├── memory_recall/              # Memory Recall task
    │   ├── {case_id}.json
    │   ├── rubric.md
    │   ├── memory_recall_eval_online.py
    │   └── memory_recall_eval_local.py
    │
    └── full_session/              # Complete session histories
        └── {case_id}_fullsession.json
```

## CPCD-Bench Tasks

CPCD-Bench evaluates models across three dimensions of long-horizon campus counseling:

### 1. Temporal-Causal Reasoning (TCR)

Analyze the temporal-causal evolution of a counselee's core distress across multiple sessions.

**Evaluation Dimensions** (0-5 scale):
- **Temporal Accuracy**: Correct chronological ordering of events
- **Causal Coherence**: Logical cause-effect relationships
- **Completeness**: Coverage of key stages (early triggers, middle amplification, late risk escalation, subtle turning points)
- **No Hallucination**: No fabricated events or characters

### 2. Session Reflection Generation (SRG)

Generate empathetic and coherent counselor responses that maintain consistency with counseling history.

**Evaluation Dimensions** (0-5 scale):
- **Empathy**: Accurate identification and acknowledgment of emotions
- **Coherence**: Consistency with history and current context
- **Professionalism**: Appropriate counseling techniques and boundaries

### 3. Long-Term Memory Recall (MR)

Accurately recall and organize relevant information from long counseling histories.

**Evaluation Dimensions** (0-5 scale):
- **Accuracy**: Factual correctness
- **Completeness**: Coverage of all key points
- **Temporal Consistency**: Correct event ordering
- **No Hallucination**: No fabricated information

## Environment Setup

```bash
# Create environment
conda create -n psy python=3.10
conda activate psy

# Install dependencies
pip install openai pandas tqdm

# Set API key (OpenRouter recommended)
export OPENROUTER_API_KEY="your_api_key"
```

## Running Evaluations

### Online Evaluation (API Generation + Scoring)

```bash
# TCR Evaluation
python eval_task_info/TCR/tcr_eval_online.py \
  --tasks "./eval_task_info/TCR" \
  --rubric "./eval_task_info/TCR/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --target-model "model/identifier" \
  --judge-model "openai/gpt-5" \
  --output "./outputs/tcr_eval.jsonl" \
  --csv-output "./outputs/tcr_eval.csv"

# SRG Evaluation
python eval_task_info/SRG/srg_eval_online.py \
  --tasks "./eval_task_info/SRG" \
  --rubric "./eval_task_info/SRG/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --target-model "model/identifier" \
  --judge-model "openai/gpt-5" \
  --output "./outputs/srg_eval.jsonl" \
  --csv-output "./outputs/srg_eval.csv"

# Memory Recall Evaluation
python eval_task_info/memory_recall/memory_recall_eval_online.py \
  --tasks "./eval_task_info/memory_recall" \
  --rubric "./eval_task_info/memory_recall/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --target-model "model/identifier" \
  --judge-model "openai/gpt-5" \
  --output "./outputs/mr_eval.jsonl" \
  --csv-output "./outputs/mr_eval.csv"
```

### Offline Evaluation (Scoring Local Responses)

```bash
# Prepare CSV with model responses (columns: task_id, model_response)
python eval_task_info/TCR/tcr_eval_local.py \
  --input-csv "./results/model_responses.csv" \
  --tasks "./eval_task_info/TCR" \
  --rubric "./eval_task_info/TCR/rubric.md" \
  --full-session-dir "./eval_task_info/full_session" \
  --judge-model "openai/gpt-5" \
  --output "./outputs/tcr_eval.jsonl" \
  --csv-output "./outputs/tcr_eval.csv"
```

## Dataset Statistics

| Component | Count | Description |
|-----------|-------|-------------|
| Student Profiles | 100 | Four-dimensional annotations |
| Dialogue Turns | ~90,000 | Semester-long trajectories |
| Text Volume | ~11.45M chars | Chinese counseling text |
| TCR Tasks | 99 | Temporal-causal reasoning cases |
| SRG Tasks | 40 | Session reflection generation cases |
| MR Tasks | 20 | Memory recall cases |

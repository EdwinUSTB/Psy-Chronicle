#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_tcr_qwen3_4b_lora_csv.py

TCR(Temporal-Causal Reasoning) 本地模型输出评分脚本。

适用场景：
  你已经用本地 Qwen3-4B LoRA 模型完成第三类 TCR 任务推理，得到 CSV，例如：
    /workspace/user-data/datasets/psy_student_eval/tcr_qwen3_4b_lora.csv

  这个脚本不会重新调用 target model。
  它只读取 CSV 中的 model_response，作为被测模型回答，
  再调用 OpenRouter 上的 judge model，例如 openai/gpt-5.2，进行评分。

本脚本按 tcr_batch_eval_runner.py 的 TCR 评测思路重构：
  - 评分指标：
      temporal_accuracy
      causal_coherence
      completeness
      no_hallucination
  - judge 输入包含：
      评测.md
      task_file
      task_id
      task_type
      case_id
      question
      input_to_model
      reference_answer
      evaluation_focus
      model_response / model_answer
      默认 full_session 完整咨询历史证据
  - 默认会把 full_session 传给 judge，用于判断时序、因果和幻觉
  - 可用 --no-full-session-in-judge 关闭
  - 支持 OpenRouter + OpenAI SDK、retry、resume、dry-run、JSONL/CSV 双输出

默认路径：
  --input-csv /workspace/user-data/datasets/psy_student_eval/tcr_qwen3_4b_lora.csv
  --tasks /workspace/user-data/datasets/psy_student_eval/TCR
  --rubric /workspace/user-data/datasets/psy_student_eval/评测.md
  --full-session-dir /workspace/user-data/datasets/psy_student_eval/full_seesion
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.2"

DEFAULT_INPUT_CSV = "/workspace/user-data/datasets/psy_student_eval/tcr_qwen3_4b_lora.csv"
DEFAULT_TASKS = "/workspace/user-data/datasets/psy_student_eval/TCR"
DEFAULT_RUBRIC = "/workspace/user-data/datasets/psy_student_eval/评测.md"
DEFAULT_FULL_SESSION_DIR = "/workspace/user-data/datasets/psy_student_eval/full_seesion"
DEFAULT_OUTPUT = "/workspace/user-data/datasets/psy_student_eval/tcr_qwen3_4b_lora_eval.jsonl"
DEFAULT_CSV_OUTPUT = "/workspace/user-data/datasets/psy_student_eval/tcr_qwen3_4b_lora_eval.csv"

SCORE_KEYS = ["temporal_accuracy", "causal_coherence", "completeness", "no_hallucination"]


@dataclass
class RunnerConfig:
    input_csv_path: Path
    tasks_path: Path
    rubric_path: Path
    full_session_dir: Optional[Path]
    full_session_file: Optional[Path]
    output_path: Path
    csv_output_path: Optional[Path]
    target_model_label: str
    judge_model: str
    judge_temperature: float
    max_judge_tokens: int
    api_key_env: str
    referer: Optional[str]
    title: Optional[str]
    dry_run: bool
    resume: bool
    task_ids: Optional[List[str]]
    case_files: Optional[List[str]]
    limit: Optional[int]
    sleep_seconds: float
    include_full_session_in_judge: bool
    max_full_session_chars: int
    skip_missing_csv_rows: bool
    skip_non_ok_rows: bool
    print_prompt: bool


@dataclass
class TaskItem:
    task: Dict[str, Any]
    task_file: Path
    task_index_in_file: int


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl_record(jsonl_path: Path, record: Dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def normalize_case_id(value: str) -> str:
    """陈明_129 -> 陈明129; Case-001 -> Case001."""
    return re.sub(r"[\s_\-]+", "", str(value).strip())


def redact_secrets(text: str) -> str:
    text = re.sub(r"sk-or-v1-[A-Za-z0-9_\-]{16,}", "sk-or-v1-***REDACTED***", text)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{20,}", "sk-***REDACTED***", text)
    return text


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def iter_task_files(tasks_path: Path) -> List[Path]:
    """Return task JSON files. If a directory is given, only top-level *.json files are loaded."""
    p = tasks_path.expanduser()
    if not p.exists():
        raise FileNotFoundError(f"任务路径不存在：{p}")
    if p.is_file():
        if p.suffix.lower() != ".json":
            raise ValueError(f"任务文件必须是 .json：{p}")
        return [p]

    files = sorted(
        x for x in p.glob("*.json")
        if x.is_file()
        and not x.name.startswith(".")
        and "fullsession" not in normalize_case_id(x.stem).lower()
        and "full_session" not in x.stem.lower()
    )
    if not files:
        raise FileNotFoundError(f"目录中没有找到任务 JSON 文件：{p}")
    return files


def load_task_items(tasks_path: Path) -> List[TaskItem]:
    """Load all TCR tasks from a file or directory. A file may contain a JSON array or a single object."""
    items: List[TaskItem] = []
    for task_file in iter_task_files(tasks_path):
        data = load_json(task_file)
        if isinstance(data, list):
            tasks = data
        elif isinstance(data, dict):
            tasks = [data]
        else:
            raise ValueError(f"任务文件应为 JSON 数组或对象：{task_file}")

        for idx, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                raise ValueError(f"任务文件 {task_file} 第 {idx} 项不是对象。")
            if "question" not in task:
                raise ValueError(f"任务文件 {task_file} 第 {idx} 项缺少 question 字段。")
            if "task_id" not in task:
                raise ValueError(f"任务文件 {task_file} 第 {idx} 项缺少 task_id 字段。")
            items.append(TaskItem(task=task, task_file=task_file, task_index_in_file=idx))
    return items


def resolve_full_session_dir(user_dir: Optional[Path], tasks_path: Path) -> Optional[Path]:
    """
    Support both typo `full_seesion` and conventional `full_session`.
    Search order follows tcr_batch_eval_runner.py style.
    """
    candidates: List[Path] = []
    if user_dir is not None:
        candidates.append(user_dir)

    candidates.extend([Path("full_seesion"), Path("full_session")])

    t = tasks_path.expanduser()
    if t.exists() and t.is_dir():
        task_dir = t
        root_dir = t.parent
    else:
        task_dir = t.parent
        root_dir = t.parent.parent

    candidates.extend([
        root_dir / "full_seesion",
        root_dir / "full_session",
        task_dir / "full_seesion",
        task_dir / "full_session",
    ])

    seen = set()
    for d in candidates:
        d = d.expanduser()
        key = str(d.resolve()) if d.exists() else str(d)
        if key in seen:
            continue
        seen.add(key)
        if d.exists() and d.is_dir():
            return d
    return user_dir


def find_full_session_file(
    task_file: Path,
    case_id: str,
    full_session_dir: Optional[Path],
    explicit_file: Optional[Path],
    csv_full_session_file: Optional[str] = None,
) -> Path:
    """
    Resolve full-session file by filename first:
      TCR/陈明129.json -> full_seesion/陈明129_fullsession.json

    Then fallback to case_id variants:
      陈明_129 -> 陈明129_fullsession.json
    """
    if explicit_file is not None:
        explicit_file = explicit_file.expanduser()
        if explicit_file.exists():
            return explicit_file
        raise FileNotFoundError(f"指定的 full-session 文件不存在：{explicit_file}")

    if csv_full_session_file:
        p = Path(csv_full_session_file).expanduser()
        if p.exists():
            return p

    if full_session_dir is None:
        raise FileNotFoundError("未找到 full-session 目录。请传 --full-session-dir full_seesion 或 full_session。")

    d = full_session_dir.expanduser()
    file_stem = task_file.stem.strip()
    normalized_stem = normalize_case_id(file_stem)
    raw_case = str(case_id).strip()
    normalized_case = normalize_case_id(raw_case)

    base_names: List[str] = []
    for base in [file_stem, normalized_stem, raw_case, normalized_case]:
        if base and base not in base_names:
            base_names.append(base)

    candidate_names: List[str] = []
    for base in base_names:
        candidate_names.extend([
            f"{base}_fullsession.json",
            f"{base}_full_session.json",
            f"{base}.json",
        ])

    for name in candidate_names:
        p = d / name
        if p.exists():
            return p

    normalized_targets = {normalize_case_id(x) for x in base_names if x}
    matches: List[Path] = []
    for p in d.glob("*.json"):
        stem_norm = normalize_case_id(p.stem)
        has_fullsession_marker = "fullsession" in stem_norm.lower() or "full_session" in p.stem.lower()
        if not has_fullsession_marker:
            continue
        if any(target and target in stem_norm for target in normalized_targets):
            matches.append(p)

    matches = sorted(set(matches))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(str(x) for x in matches)
        raise FileNotFoundError(
            f"任务文件 {task_file.name} 找到多个可能的 full-session 文件，请检查命名或用 --full-session-file 指定：{names}"
        )

    raise FileNotFoundError(
        f"在目录 {d} 中找不到任务文件 {task_file.name} 对应的 full-session 文件。"
        f"按规范应存在：{file_stem}_fullsession.json。已尝试：{candidate_names}"
    )


def render_messages(messages: Any) -> str:
    """Render full-session list[dict] into a compact readable transcript."""
    if not isinstance(messages, list):
        return json.dumps(messages, ensure_ascii=False, indent=2)

    lines: List[str] = []
    for idx, msg in enumerate(messages, start=1):
        if isinstance(msg, dict):
            role = msg.get("role", "Unknown")
            content = msg.get("content", "")
            lines.append(f"[{idx:04d}] {role}: {content}")
        else:
            lines.append(f"[{idx:04d}] {json.dumps(msg, ensure_ascii=False)}")
    return "\n".join(lines)


def maybe_truncate(text: str, max_chars: int) -> Tuple[str, bool]:
    if max_chars and max_chars > 0 and len(text) > max_chars:
        half = max_chars // 2
        return (
            text[:half]
            + "\n\n...[中间内容因 --max-full-session-chars 被截断]...\n\n"
            + text[-half:],
            True,
        )
    return text, False


def build_judge_messages(
    task: Dict[str, Any],
    target_answer: str,
    rubric_text: str,
    full_session_text: Optional[str],
    full_session_path: Optional[Path],
    task_file: Path,
) -> List[Dict[str, str]]:
    task_brief = {
        "task_file": task_file.name,
        "task_id": task.get("task_id"),
        "task_type": task.get("task_type"),
        "case_id": task.get("case_id"),
        "question": task.get("question"),
        "input_to_model": task.get("input_to_model", {}),
        "reference_answer": task.get("reference_answer"),
        "evaluation_focus": task.get("evaluation_focus", {}),
    }

    evidence_block = ""
    if full_session_text is not None:
        evidence_block = f"""

【完整咨询历史证据】
文件：{full_session_path.name if full_session_path else "unknown"}
{full_session_text}
"""

    system = (
        "你是严格的校园心理咨询案例时序—因果推理评测员。"
        "你需要根据评分规则、reference_answer、evaluation_focus 和咨询历史，评估被测模型回答。"
        "重点关注：事件时间顺序是否正确、因果链条是否合理、关键阶段是否完整、是否编造不存在的信息。"
        "不要因为答案风格和参考答案不同就扣分；只在事实错误、顺序混乱、因果断裂、关键遗漏或幻觉时扣分。"
        "必须只输出 JSON，不要输出 Markdown 或额外解释。"
    )

    user = f"""请评测下面的被测模型回答。

【评分规则】
{rubric_text}

【任务信息】
{json.dumps(task_brief, ensure_ascii=False, indent=2)}
{evidence_block}

【被测模型回答】
{target_answer}

请严格输出以下 JSON 结构，四项分数必须是 0 到 5 的整数：
{{
  "scores": {{
    "temporal_accuracy": 0,
    "causal_coherence": 0,
    "completeness": 0,
    "no_hallucination": 0
  }},
  "rationales": {{
    "temporal_accuracy": "...",
    "causal_coherence": "...",
    "completeness": "...",
    "no_hallucination": "..."
  }},
  "overall_comment": "..."
}}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("无法从评测模型输出中解析 JSON。")


def normalize_judge_result(raw_text: str) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text)
    scores = parsed.get("scores", {})
    normalized_scores: Dict[str, int] = {}
    alternatives_map = {
        "temporal_accuracy": ["Temporal Accuracy", "temporal accuracy", "TemporalAccuracy"],
        "causal_coherence": ["Causal Coherence", "causal coherence", "CausalCoherence"],
        "completeness": ["Completeness"],
        "no_hallucination": ["No Hallucination", "no hallucination", "NoHallucination"],
    }

    for key in SCORE_KEYS:
        val = scores.get(key)
        if val is None:
            for alt in alternatives_map[key]:
                if alt in scores:
                    val = scores[alt]
                    break

        try:
            iv = int(val)
        except Exception as exc:
            raise ValueError(f"评分字段 {key} 缺失或不是整数：{val!r}") from exc

        if not (0 <= iv <= 5):
            raise ValueError(f"评分字段 {key} 超出 0-5：{iv}")

        normalized_scores[key] = iv

    parsed["scores"] = normalized_scores
    parsed["average_score"] = round(sum(normalized_scores.values()) / len(SCORE_KEYS), 3)
    return parsed


def get_openrouter_client(config: RunnerConfig):
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"未找到环境变量 {config.api_key_env}。请先设置：export {config.api_key_env}='sk-or-v1-...'"
        )

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai 包。请运行：pip install openai") from exc

    headers = {}
    if config.referer:
        headers["HTTP-Referer"] = config.referer
    if config.title:
        headers["X-OpenRouter-Title"] = config.title

    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, default_headers=headers or None)


def call_chat_completion(
    client: Any,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    retries: int = 2,
) -> str:
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(2**attempt)
            else:
                break

    raise RuntimeError(f"模型调用失败：{last_error}")


def composite_run_id(item: TaskItem) -> str:
    tid = item.task.get("task_id")
    if tid:
        return str(tid)
    return f"{item.task_file.stem}#{item.task_index_in_file}"


def load_existing_task_ids(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.exists():
        return done

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only skip successful records. Failed records should be retryable on resume.
            if obj.get("error"):
                continue

            run_id = obj.get("run_id") or obj.get("task_id")
            if run_id:
                done.add(str(run_id))

    return done


def select_task_items(
    items: Sequence[TaskItem],
    task_ids: Optional[List[str]],
    case_files: Optional[List[str]],
    limit: Optional[int],
) -> List[TaskItem]:
    selected = list(items)

    if task_ids:
        allow = set(task_ids)
        selected = [i for i in selected if str(i.task.get("task_id")) in allow]

    if case_files:
        allow_stems = {Path(x).stem for x in case_files}
        selected = [i for i in selected if i.task_file.stem in allow_stems or i.task_file.name in case_files]

    if limit is not None:
        selected = selected[:limit]

    return selected


def build_csv_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if task_id:
            index[task_id] = row

    return index


def append_csv(csv_path: Path, records: List[Dict[str, Any]]) -> None:
    if not records:
        return

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "task_file",
        "task_id",
        "case_id",
        "question",
        "target_model",
        "judge_model",
        "temporal_accuracy",
        "causal_coherence",
        "completeness",
        "no_hallucination",
        "average_score",
        "temporal_accuracy_rationale",
        "causal_coherence_rationale",
        "completeness_rationale",
        "no_hallucination_rationale",
        "model_answer",
        "overall_comment",
        "full_session_file",
        "full_session_truncated",
        "error",
    ]

    exists = csv_path.exists() and csv_path.stat().st_size > 0

    with csv_path.open("a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not exists:
            writer.writeheader()

        for rec in records:
            judge = rec.get("judge_result") or {}
            scores = judge.get("scores") or {}
            rationales = judge.get("rationales") or judge.get("rationale") or {}

            writer.writerow(
                {
                    "run_id": rec.get("run_id"),
                    "task_file": rec.get("task_file"),
                    "task_id": rec.get("task_id"),
                    "case_id": rec.get("case_id"),
                    "question": rec.get("question"),
                    "target_model": rec.get("target_model"),
                    "judge_model": rec.get("judge_model"),
                    "temporal_accuracy": scores.get("temporal_accuracy"),
                    "causal_coherence": scores.get("causal_coherence"),
                    "completeness": scores.get("completeness"),
                    "no_hallucination": scores.get("no_hallucination"),
                    "average_score": judge.get("average_score"),
                    "temporal_accuracy_rationale": rationales.get("temporal_accuracy"),
                    "causal_coherence_rationale": rationales.get("causal_coherence"),
                    "completeness_rationale": rationales.get("completeness"),
                    "no_hallucination_rationale": rationales.get("no_hallucination"),
                    "model_answer": rec.get("model_answer"),
                    "overall_comment": judge.get("overall_comment"),
                    "full_session_file": rec.get("full_session_file"),
                    "full_session_truncated": rec.get("full_session_truncated"),
                    "error": rec.get("error"),
                }
            )


def run(config: RunnerConfig) -> None:
    csv_rows = read_csv_rows(config.input_csv_path)
    csv_index = build_csv_index(csv_rows)

    items = load_task_items(config.tasks_path)
    items = select_task_items(items, config.task_ids, config.case_files, config.limit)
    rubric_text = redact_secrets(read_text(config.rubric_path))

    if config.skip_missing_csv_rows:
        items = [i for i in items if composite_run_id(i) in csv_index]

    if config.resume:
        done_ids = load_existing_task_ids(config.output_path)
        items = [i for i in items if composite_run_id(i) not in done_ids]
    else:
        if config.output_path.exists() and not config.dry_run:
            config.output_path.unlink()
        if config.csv_output_path and config.csv_output_path.exists() and not config.dry_run:
            config.csv_output_path.unlink()

    full_session_dir = resolve_full_session_dir(config.full_session_dir, config.tasks_path)
    client = None if config.dry_run else get_openrouter_client(config)

    print(f"Loaded {len(csv_rows)} CSV row(s) from: {config.input_csv_path}", flush=True)
    print(f"Loaded {len(items)} TCR task(s) to evaluate from: {config.tasks_path}", flush=True)

    if full_session_dir:
        print(f"Using full-session directory: {full_session_dir}", flush=True)

    for idx, item in enumerate(items, start=1):
        task = item.task
        task_id = str(task.get("task_id", f"{item.task_file.stem}#{item.task_index_in_file}"))
        run_id = composite_run_id(item)
        case_id = str(task.get("case_id") or item.task_file.stem)
        csv_row = csv_index.get(run_id)

        print(f"[{idx}/{len(items)}] Grading {item.task_file.name} :: {task_id} ...", flush=True)

        try:
            if csv_row is None:
                raise ValueError(f"CSV 中找不到 task_id={run_id} 的模型输出。")

            row_status = str(csv_row.get("status", "")).strip().lower()
            if config.skip_non_ok_rows and row_status and row_status != "ok":
                raise ValueError(f"CSV 行 status={row_status!r}，不是 ok。")

            model_answer = str(csv_row.get("model_response", "") or "").strip()
            if not model_answer:
                raise ValueError("CSV 中 model_response 为空。")

            full_session_path = find_full_session_file(
                task_file=item.task_file,
                case_id=case_id,
                full_session_dir=full_session_dir,
                explicit_file=config.full_session_file,
                csv_full_session_file=csv_row.get("full_session_file"),
            )

            full_session_obj = load_json(full_session_path)
            full_session_text = render_messages(full_session_obj)
            full_session_text_for_judge, truncated_for_judge = maybe_truncate(
                full_session_text, config.max_full_session_chars
            )

            judge_full_session_text = full_session_text_for_judge if config.include_full_session_in_judge else None
            judge_messages = build_judge_messages(
                task=task,
                target_answer=model_answer,
                rubric_text=rubric_text,
                full_session_text=judge_full_session_text,
                full_session_path=full_session_path if config.include_full_session_in_judge else None,
                task_file=item.task_file,
            )

            if config.print_prompt:
                print("=== Judge Prompt ===")
                print(json.dumps(judge_messages, ensure_ascii=False, indent=2)[:20000])

            if config.dry_run:
                raw_judge = json.dumps(
                    {
                        "scores": {
                            "temporal_accuracy": 0,
                            "causal_coherence": 0,
                            "completeness": 0,
                            "no_hallucination": 0,
                        },
                        "rationales": {
                            "temporal_accuracy": "dry-run 未调用评测模型",
                            "causal_coherence": "dry-run 未调用评测模型",
                            "completeness": "dry-run 未调用评测模型",
                            "no_hallucination": "dry-run 未调用评测模型",
                        },
                        "overall_comment": "dry-run preview",
                    },
                    ensure_ascii=False,
                )
                print("=== Dry-run judge prompt preview ===")
                print(json.dumps(judge_messages, ensure_ascii=False, indent=2)[:5000])
            else:
                raw_judge = call_chat_completion(
                    client=client,
                    model=config.judge_model,
                    messages=judge_messages,
                    temperature=config.judge_temperature,
                    max_tokens=config.max_judge_tokens,
                )

            judge_result = normalize_judge_result(raw_judge)

            record: Dict[str, Any] = {
                "run_id": run_id,
                "task_file": str(item.task_file),
                "task_index_in_file": item.task_index_in_file,
                "task_id": task_id,
                "task_type": task.get("task_type"),
                "case_id": case_id,
                "question": task.get("question"),
                "input_to_model": task.get("input_to_model", {}),
                "reference_answer": task.get("reference_answer"),
                "evaluation_focus": task.get("evaluation_focus"),
                "full_session_file": str(full_session_path),
                "full_session_truncated": truncated_for_judge,
                "target_model": config.target_model_label,
                "judge_model": config.judge_model,
                "model_answer": model_answer,
                "judge_result": judge_result,
                "raw_judge_output": raw_judge,
                "source_csv": str(config.input_csv_path),
                "source_csv_row": csv_row,
            }

            avg = judge_result.get("average_score")
            print(f"    完成：平均分={avg}", flush=True)

        except Exception as exc:
            record = {
                "run_id": run_id,
                "task_file": str(item.task_file),
                "task_index_in_file": item.task_index_in_file,
                "task_id": task_id,
                "task_type": task.get("task_type"),
                "case_id": case_id,
                "question": task.get("question"),
                "target_model": config.target_model_label,
                "judge_model": config.judge_model,
                "source_csv": str(config.input_csv_path),
                "error": str(exc),
            }
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)

        if not config.dry_run:
            write_jsonl_record(config.output_path, record)
            if config.csv_output_path:
                append_csv(config.csv_output_path, [record])
        else:
            print(json.dumps(record, ensure_ascii=False, indent=2)[:5000])

        if config.sleep_seconds > 0 and idx < len(items):
            time.sleep(config.sleep_seconds)

    if config.dry_run:
        print("\nDry-run finished. No files were written.")
    else:
        print(f"\nDone. JSONL saved to: {config.output_path}")
        if config.csv_output_path:
            print(f"CSV saved to: {config.csv_output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> RunnerConfig:
    p = argparse.ArgumentParser(description="Grade TCR CSV outputs through OpenRouter/OpenAI SDK.")
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="本地模型 TCR 输出 CSV，例如 tcr_qwen3_4b_lora.csv")
    p.add_argument("--tasks", default=DEFAULT_TASKS, help="TCR 任务 JSON 文件或目录。例如 TCR；目录下每个 *.json 都会被读取。")
    p.add_argument("--rubric", default=DEFAULT_RUBRIC, help="评分规则 Markdown 文件，例如 评测.md")
    p.add_argument("--full-session-dir", default=DEFAULT_FULL_SESSION_DIR, help="完整历史目录，例如 full_session 或 full_seesion")
    p.add_argument("--full-session-file", default=None, help="显式指定完整历史文件。通常只用于单个任务文件；目录批处理时不建议使用。")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="JSONL 输出路径")
    p.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, help="CSV 输出路径；传空字符串可关闭")

    p.add_argument("--target-model-label", default="local/qwen3-4b-lora", help="输出文件中的 target_model 标签；不会用于 API 调用")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="评测模型，例如 openai/gpt-5.2")
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--max-judge-tokens", type=int, default=2200)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--referer", default=os.environ.get("OPENROUTER_HTTP_REFERER"), help="OpenRouter 可选 HTTP-Referer")
    p.add_argument("--title", default=os.environ.get("OPENROUTER_APP_TITLE", "Counseling TCR CSV Eval"), help="OpenRouter 可选 X-OpenRouter-Title")
    p.add_argument("--dry-run", action="store_true", help="只检查文件解析和 prompt 构建，不调用模型、不写结果")
    p.add_argument("--print-prompt", action="store_true", help="逐条打印发送给 judge 的 prompt")
    p.add_argument("--resume", action="store_true", help="跳过输出 JSONL 中已有且无 error 的 run_id/task_id")
    p.add_argument("--task-id", action="append", dest="task_ids", help="只运行指定 task_id；可重复传入")
    p.add_argument("--case-file", action="append", dest="case_files", help="只运行指定任务文件名或 stem，例如 陈明129 或 陈明129.json；可重复传入")
    p.add_argument("--limit", type=int, default=None, help="只运行前 N 条任务")
    p.add_argument("--sleep-seconds", type=float, default=0.0, help="任务之间暂停秒数，用于限速")
    p.add_argument("--no-full-session-in-judge", action="store_true", help="评测模型不接收完整历史，只接收 reference_answer/evaluation_focus/被测回答。默认会接收完整历史以判断幻觉。")
    p.add_argument("--max-full-session-chars", type=int, default=0, help="限制传给 judge 的完整历史字符数；0 表示不截断。")
    p.add_argument("--no-skip-missing-csv-rows", action="store_true", help="默认只评 CSV 中存在的 task_id；加此参数后缺失行会记录 error。")
    p.add_argument("--include-non-ok-rows", action="store_true", help="默认跳过/报错 CSV 中 status 非 ok 的行；加此参数后仍尝试评分。")

    args = p.parse_args(argv)

    csv_output: Optional[Path]
    if args.csv_output == "":
        csv_output = None
    else:
        csv_output = Path(args.csv_output)

    return RunnerConfig(
        input_csv_path=Path(args.input_csv),
        tasks_path=Path(args.tasks),
        rubric_path=Path(args.rubric),
        full_session_dir=Path(args.full_session_dir) if args.full_session_dir else None,
        full_session_file=Path(args.full_session_file) if args.full_session_file else None,
        output_path=Path(args.output),
        csv_output_path=csv_output,
        target_model_label=args.target_model_label,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
        max_judge_tokens=args.max_judge_tokens,
        api_key_env=args.api_key_env,
        referer=args.referer,
        title=args.title,
        dry_run=args.dry_run,
        resume=args.resume,
        task_ids=args.task_ids,
        case_files=args.case_files,
        limit=args.limit,
        sleep_seconds=args.sleep_seconds,
        include_full_session_in_judge=not args.no_full_session_in_judge,
        max_full_session_chars=args.max_full_session_chars,
        skip_missing_csv_rows=not args.no_skip_missing_csv_rows,
        skip_non_ok_rows=not args.include_non_ok_rows,
        print_prompt=args.print_prompt,
    )


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()


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
DEFAULT_TARGET_MODEL = "openai/gpt-5.2"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.2"
SCORE_KEYS = ["accuracy", "completeness", "temporal_consistency", "no_hallucination"]


@dataclass
class RunnerConfig:
    tasks_path: Path
    rubric_path: Path
    full_session_dir: Optional[Path]
    full_session_file: Optional[Path]
    output_path: Path
    csv_output_path: Optional[Path]
    target_model: str
    judge_model: str
    target_temperature: float
    judge_temperature: float
    max_target_tokens: int
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


@dataclass
class TaskItem:
    task: Dict[str, Any]
    task_file: Path
    task_index_in_file: int


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_case_id(value: str) -> str:
    """陈明_129 -> 陈明129; Case-001 -> Case001."""
    return re.sub(r"[\s_\-]+", "", str(value).strip())


def safe_slug(value: str) -> str:
    """Make a stable fallback id fragment for resume/composite ids."""
    value = normalize_case_id(value)
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", value) or "unknown"


def redact_secrets(text: str) -> str:
    # OpenRouter keys normally start with sk-or-v1-. Keep enough shape for debugging without leaking.
    text = re.sub(r"sk-or-v1-[A-Za-z0-9_\-]{16,}", "sk-or-v1-***REDACTED***", text)
    text = re.sub(r"sk-[A-Za-z0-9_\-]{20,}", "sk-***REDACTED***", text)
    return text


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
    """Load all tasks from a file or directory. A file may contain a JSON array or a single object."""
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
            items.append(TaskItem(task=task, task_file=task_file, task_index_in_file=idx))
    return items


def default_output_path(tasks_path: Path, suffix: str) -> Path:
    """Put default outputs in current working directory with a stable name."""
    if tasks_path.expanduser().is_dir():
        return Path(f"memory_recall_batch_results{suffix}")
    return Path(f"{tasks_path.stem}_memory_recall_results{suffix}")


def resolve_full_session_dir(user_dir: Optional[Path], tasks_path: Path) -> Optional[Path]:
    """
    Support both requested typo `full_seesion` and conventional `full_session`.

    Search order:
    1. Explicit --full-session-dir
    2. ./full_seesion, ./full_session from current working directory
    3. Sibling directories next to the task directory, e.g. project_root/full_seesion
    4. Directories under the task directory itself, for unusual layouts
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
) -> Path:
    """
    Resolve full-session file by filename first:
      memory recall/陈明129.json -> full_seesion/陈明129_fullsession.json

    Then fallback to case_id variants:
      陈明_129 -> 陈明129_fullsession.json
    """
    if explicit_file is not None:
        explicit_file = explicit_file.expanduser()
        if explicit_file.exists():
            return explicit_file
        raise FileNotFoundError(f"指定的 full-session 文件不存在：{explicit_file}")

    if full_session_dir is None:
        raise FileNotFoundError("未找到 full-session 目录。请传 --full-session-dir full_seesion。")

    d = full_session_dir.expanduser()
    file_stem = task_file.stem.strip()
    normalized_stem = normalize_case_id(file_stem)
    raw_case = str(case_id).strip()
    normalized_case = normalize_case_id(raw_case)

    base_names = []
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

    # Fallback: search any fullsession file with matching normalized stem/case.
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


def build_target_messages(task: Dict[str, Any], full_session_text: str, full_session_path: Path, task_file: Path) -> List[Dict[str, str]]:
    question = task.get("question", "")
    task_for_prompt = {
        "task_id": task.get("task_id"),
        "task_type": task.get("task_type"),
        "case_id": task.get("case_id"),
        "input_to_model": task.get("input_to_model", {}),
    }
    system = (
        "Please answer questions based solely on the complete consultation history provided. Do not fabricate facts, people, figures, or causal relationships not present in the materials. If there is insufficient evidence in the materials, please clearly state 'Uncertain/Not explicitly mentioned in the materials.'"
    )
    user = f"""Please answer the questions based on the following consultation history.

【task file】
{task_file.name}

【task metadata】
{json.dumps(task_for_prompt, ensure_ascii=False, indent=2)}

【Complete consultation history documents】
{full_session_path.name}

【Complete consultation history】
{full_session_text}

【question】
{question}

Please provide a concise and accurate answer based on the above information."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


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
        "answer_source": task.get("answer_source", {}),
    }
    evidence_block = ""
    if full_session_text is not None:
        evidence_block = f"""

【完整咨询历史证据】
文件：{full_session_path.name if full_session_path else "unknown"}
{full_session_text}
"""

    system = (
        "你是严格的事实回忆评测员。你需要根据评分规则，评估被测模型回答是否准确回忆咨询历史。"
        "请特别关注事实准确性、覆盖关键点、时间顺序、以及是否编造不存在的信息。"
        "不要因为答案简短就自动扣分；只在关键点缺失、事实错误、顺序混乱或编造时扣分。"
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
    "accuracy": 0,
    "completeness": 0,
    "temporal_consistency": 0,
    "no_hallucination": 0
  }},
  "rationales": {{
    "accuracy": "...",
    "completeness": "...",
    "temporal_consistency": "...",
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
        "accuracy": ["Accuracy"],
        "completeness": ["Completeness"],
        "temporal_consistency": ["Temporal Consistency", "temporal consistency", "TemporalConsistency"],
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
        raise RuntimeError("缺少 openai 包。请运行：pip install -r requirements_memory_eval.txt") from exc

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
        except Exception as exc:  # noqa: BLE001
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
    done = set()
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


def write_jsonl_record(jsonl_path: Path, record: Dict[str, Any]) -> None:
    """Append one result record to a JSONL file, creating parent directories if needed."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


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
        "accuracy",
        "completeness",
        "temporal_consistency",
        "no_hallucination",
        "average_score",
        "model_answer",
        "overall_comment",
        "full_session_file",
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
            writer.writerow(
                {
                    "run_id": rec.get("run_id"),
                    "task_file": rec.get("task_file"),
                    "task_id": rec.get("task_id"),
                    "case_id": rec.get("case_id"),
                    "question": rec.get("question"),
                    "target_model": rec.get("target_model"),
                    "judge_model": rec.get("judge_model"),
                    "accuracy": scores.get("accuracy"),
                    "completeness": scores.get("completeness"),
                    "temporal_consistency": scores.get("temporal_consistency"),
                    "no_hallucination": scores.get("no_hallucination"),
                    "average_score": judge.get("average_score"),
                    "model_answer": rec.get("model_answer"),
                    "overall_comment": judge.get("overall_comment"),
                    "full_session_file": rec.get("full_session_file"),
                    "error": rec.get("error"),
                }
            )


def run(config: RunnerConfig) -> None:
    items = load_task_items(config.tasks_path)
    items = select_task_items(items, config.task_ids, config.case_files, config.limit)
    rubric_text = redact_secrets(read_text(config.rubric_path))

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

    print(f"Loaded {len(items)} task(s) from: {config.tasks_path}", flush=True)
    if full_session_dir:
        print(f"Using full-session directory: {full_session_dir}", flush=True)

    for idx, item in enumerate(items, start=1):
        task = item.task
        task_id = str(task.get("task_id", f"{item.task_file.stem}#{item.task_index_in_file}"))
        run_id = composite_run_id(item)
        case_id = str(task.get("case_id") or item.task_file.stem)
        print(f"[{idx}/{len(items)}] Running {item.task_file.name} :: {task_id} ...", flush=True)

        try:
            full_session_path = find_full_session_file(
                task_file=item.task_file,
                case_id=case_id,
                full_session_dir=full_session_dir,
                explicit_file=config.full_session_file,
            )
            full_session_obj = load_json(full_session_path)
            full_session_text = render_messages(full_session_obj)
            full_session_text_for_model, truncated_for_model = maybe_truncate(
                full_session_text, config.max_full_session_chars
            )

            target_messages = build_target_messages(task, full_session_text_for_model, full_session_path, item.task_file)
            if config.dry_run:
                model_answer = "[DRY RUN] 这里会是被测模型根据完整咨询历史和 question 生成的回答。"
            else:
                model_answer = call_chat_completion(
                    client=client,
                    model=config.target_model,
                    messages=target_messages,
                    temperature=config.target_temperature,
                    max_tokens=config.max_target_tokens,
                )

            judge_full_session_text = full_session_text_for_model if config.include_full_session_in_judge else None
            judge_messages = build_judge_messages(
                task=task,
                target_answer=model_answer,
                rubric_text=rubric_text,
                full_session_text=judge_full_session_text,
                full_session_path=full_session_path if config.include_full_session_in_judge else None,
                task_file=item.task_file,
            )
            if config.dry_run:
                raw_judge = json.dumps(
                    {
                        "scores": {
                            "accuracy": 0,
                            "completeness": 0,
                            "temporal_consistency": 0,
                            "no_hallucination": 0,
                        },
                        "rationales": {
                            "accuracy": "dry-run 未调用评测模型",
                            "completeness": "dry-run 未调用评测模型",
                            "temporal_consistency": "dry-run 未调用评测模型",
                            "no_hallucination": "dry-run 未调用评测模型",
                        },
                        "overall_comment": "dry-run preview",
                    },
                    ensure_ascii=False,
                )
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
                "answer_source": task.get("answer_source"),
                "full_session_file": str(full_session_path),
                "full_session_truncated": truncated_for_model,
                "target_model": config.target_model,
                "judge_model": config.judge_model,
                "model_answer": model_answer,
                "judge_result": judge_result,
                "raw_judge_output": raw_judge,
            }
        except Exception as exc:  # noqa: BLE001 - keep processing next tasks/files
            record = {
                "run_id": run_id,
                "task_file": str(item.task_file),
                "task_index_in_file": item.task_index_in_file,
                "task_id": task_id,
                "task_type": task.get("task_type"),
                "case_id": case_id,
                "question": task.get("question"),
                "target_model": config.target_model,
                "judge_model": config.judge_model,
                "error": str(exc),
            }
            print(f"  ERROR: {exc}", file=sys.stderr, flush=True)

        if not config.dry_run:
            write_jsonl_record(config.output_path, record)
            if config.csv_output_path:
                append_csv(config.csv_output_path, [record])
        else:
            print(json.dumps(record, ensure_ascii=False, indent=2)[:4000])

        if config.sleep_seconds > 0 and idx < len(items):
            time.sleep(config.sleep_seconds)

    if config.dry_run:
        print("\nDry-run finished. No files were written.")
    else:
        print(f"\nDone. JSONL saved to: {config.output_path}")
        if config.csv_output_path:
            print(f"CSV saved to: {config.csv_output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> RunnerConfig:
    p = argparse.ArgumentParser(description="Run batch memory-recall evaluation tasks through OpenRouter/OpenAI SDK.")
    p.add_argument(
        "--tasks",
        required=True,
        help="任务 JSON 文件或目录。例如 'memory recall'；目录下每个 *.json 都会被读取。",
    )
    p.add_argument("--rubric", required=True, help="评分规则 Markdown 文件，例如 评分.md")
    p.add_argument(
        "--full-session-dir",
        default=None,
        help="完整历史目录，例如 full_seesion。默认自动查找 ./full_seesion、./full_session 以及任务目录的兄弟目录。",
    )
    p.add_argument(
        "--full-session-file",
        default=None,
        help="显式指定完整历史文件。通常只用于单个任务文件；目录批处理时不建议使用。",
    )
    p.add_argument("--output", default=None, help="JSONL 输出路径；默认按任务路径自动命名")
    p.add_argument("--csv-output", default=None, help="CSV 输出路径；传空字符串可关闭；默认按任务路径自动命名")
    p.add_argument("--target-model", default=DEFAULT_TARGET_MODEL, help="被测模型，例如 openai/gpt-5.2")
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="评测模型，例如 openai/gpt-5.2")
    p.add_argument("--target-temperature", type=float, default=0.0)
    p.add_argument("--judge-temperature", type=float, default=0.0)
    p.add_argument("--max-target-tokens", type=int, default=512)
    p.add_argument("--max-judge-tokens", type=int, default=1200)
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--referer", default=os.environ.get("OPENROUTER_HTTP_REFERER"), help="OpenRouter 可选 HTTP-Referer")
    p.add_argument("--title", default=os.environ.get("OPENROUTER_APP_TITLE", "Counseling Memory Recall Batch Eval"), help="OpenRouter 可选 X-OpenRouter-Title")
    p.add_argument("--dry-run", action="store_true", help="只检查文件解析和 prompt 构建，不调用模型、不写结果")
    p.add_argument("--resume", action="store_true", help="跳过输出 JSONL 中已有 run_id/task_id")
    p.add_argument("--task-id", action="append", dest="task_ids", help="只运行指定 task_id；可重复传入")
    p.add_argument("--case-file", action="append", dest="case_files", help="只运行指定任务文件名或 stem，例如 陈明129 或 陈明129.json；可重复传入")
    p.add_argument("--limit", type=int, default=None, help="只运行前 N 条任务")
    p.add_argument("--sleep-seconds", type=float, default=0.0, help="任务之间暂停秒数，用于限速")
    p.add_argument(
        "--no-full-session-in-judge",
        action="store_true",
        help="评测模型不接收完整历史，只接收 reference_answer/answer_source/被测回答。默认会接收完整历史以判断幻觉。",
    )
    p.add_argument(
        "--max-full-session-chars",
        type=int,
        default=0,
        help="限制传给模型的完整历史字符数；0 表示不截断。上下文很长且模型窗口不足时可设置。",
    )

    args = p.parse_args(argv)
    tasks_path = Path(args.tasks)
    output = Path(args.output) if args.output else default_output_path(tasks_path, ".jsonl")
    if args.csv_output == "":
        csv_output = None
    elif args.csv_output is None:
        csv_output = default_output_path(tasks_path, ".csv")
    else:
        csv_output = Path(args.csv_output)

    return RunnerConfig(
        tasks_path=tasks_path,
        rubric_path=Path(args.rubric),
        full_session_dir=Path(args.full_session_dir) if args.full_session_dir else None,
        full_session_file=Path(args.full_session_file) if args.full_session_file else None,
        output_path=output,
        csv_output_path=csv_output,
        target_model=args.target_model,
        judge_model=args.judge_model,
        target_temperature=args.target_temperature,
        judge_temperature=args.judge_temperature,
        max_target_tokens=args.max_target_tokens,
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
    )


def main() -> None:
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()

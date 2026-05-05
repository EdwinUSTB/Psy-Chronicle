#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_srg_qwen3_4b_lora_csv.py

Evaluate SRG local-model CSV outputs with OpenRouter judge model.

This follows eval_runner_all.py's SRG judging logic, but does NOT call the
target model. It reads model_response from a local inference CSV, then sends it
to the judge model together with task context, evaluation_focus, and rubric.

Default judge input:
  rubric + task_id + representative_point + student_profile_summary
  + history_until_previous_session + current_session_event
  + current_session_context + current_student_utterance
  + evaluation_focus + model_response_to_evaluate

reference_answer is NOT sent by default. Add --include-reference-answer to send it.
full_session is not used for SRG, consistent with eval_runner_all.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_JUDGE_MODEL = "openai/gpt-5.2"
DEFAULT_INPUT_CSV = "/workspace/user-data/datasets/psy_student_eval/srg_qwen3_4b_lora.csv"
DEFAULT_TASKS = "/workspace/user-data/datasets/psy_student_eval/session"
DEFAULT_RUBRIC = "/workspace/user-data/datasets/psy_student_eval/评测标准.md"
DEFAULT_OUTPUT = "/workspace/user-data/datasets/psy_student_eval/srg_qwen3_4b_lora_eval.jsonl"
DEFAULT_CSV_OUTPUT = "/workspace/user-data/datasets/psy_student_eval/srg_qwen3_4b_lora_eval.csv"

METRICS = ("empathy", "coherence", "professionalism")
SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"(?im)^\s*psy\s*api\s*[:：].*$"),
    re.compile(r"(?im)^\s*(api\s*key|openrouter_api_key|OPENROUTER_API_KEY)\s*[:=：].*$"),
]


@dataclass
class Config:
    input_csv: Path
    tasks: Path
    rubric: Path
    output: Path
    csv_output: Optional[Path]
    target_model_label: str
    judge_model: str
    base_url: str
    api_key: str
    site_url: Optional[str]
    app_title: Optional[str]
    judge_temperature: float
    judge_max_tokens: int
    timeout: float
    max_retries: int
    retry_base_sleep: float
    sleep_between_tasks: float
    task_ids: Optional[List[str]]
    case_files: Optional[List[str]]
    limit: Optional[int]
    resume: bool
    dry_run: bool
    print_prompt: bool
    include_reference_answer: bool
    skip_missing_csv_rows: bool
    skip_non_ok_rows: bool


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


def sanitize_text(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_single_task_file(path: Path) -> List[Dict[str, Any]]:
    raw = read_text(path)
    if not raw.strip():
        print(f"[警告] 跳过空文件: {path}")
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"解析出错！文件 {path} 不是合法 JSON：{e}") from e

    if isinstance(data, list):
        tasks = data
    elif isinstance(data, dict):
        for key in ("tasks", "data", "items"):
            if isinstance(data.get(key), list):
                tasks = data[key]
                break
        else:
            tasks = [data] if "task_id" in data else None
            if tasks is None:
                raise ValueError(f"文件 {path} 任务 JSON 必须是 list，或包含 tasks/data/items。")
    else:
        raise ValueError(f"文件 {path} 任务 JSON 顶层必须是 list 或 dict。")

    out = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"文件 {path} 第 {idx} 个任务不是对象。")
        if "task_id" not in task:
            raise ValueError(f"文件 {path} 第 {idx} 个任务缺少 task_id。")
        if "input_to_model" not in task or not isinstance(task["input_to_model"], dict):
            raise ValueError(f"文件 {path} 任务 {task.get('task_id', idx)} 缺少 input_to_model 对象。")
        if "evaluation_focus" not in task or not isinstance(task["evaluation_focus"], dict):
            raise ValueError(f"文件 {path} 任务 {task.get('task_id', idx)} 缺少 evaluation_focus 对象。")
        out.append(task)
    return out


def iter_task_files(path: Path) -> List[Path]:
    p = path.expanduser()
    if not p.exists():
        raise FileNotFoundError(f"任务路径不存在：{p}")
    if p.is_file():
        if p.suffix.lower() != ".json":
            raise ValueError(f"任务文件必须是 .json：{p}")
        return [p]
    files = sorted(
        x for x in p.glob("*.json")
        if x.is_file() and not x.name.startswith(".")
        and "fullsession" not in x.stem.lower()
        and "full_session" not in x.stem.lower()
    )
    if not files:
        raise FileNotFoundError(f"目录中没有找到任务 JSON 文件：{p}")
    return files


def load_task_items(path: Path) -> List[TaskItem]:
    items: List[TaskItem] = []
    for fp in iter_task_files(path):
        for idx, task in enumerate(load_single_task_file(fp), start=1):
            items.append(TaskItem(task=task, task_file=fp, task_index_in_file=idx))
    return items


def format_context(context: Any) -> str:
    if not context:
        return "无"
    if not isinstance(context, list):
        return str(context)
    lines = []
    for turn in context:
        if isinstance(turn, dict):
            role = turn.get("role", "Unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(turn))
    return "\n".join(lines)


def build_judge_messages(task: Dict[str, Any], model_response: str, rubric_text: str, include_reference_answer: bool) -> List[Dict[str, str]]:
    inp = task["input_to_model"]
    reference_block = ""
    if include_reference_answer and task.get("reference_answer"):
        reference_block = f"""
【reference_answer，高分参考方向；不要要求逐字相似】
{task.get('reference_answer')}
""".strip()

    system_prompt = (
        "你是严格、审慎的心理咨询回复评测员。请基于评分标准、任务 evaluation_focus、"
        "学生画像、历史上下文和模型回复，对模型回复进行评分。\n\n"
        "评分规则：\n"
        "1. 分别给 empathy、coherence、professionalism 三项打 1-5 的整数分。\n"
        "2. 必须同时参考通用评分标准和该任务的 evaluation_focus。\n"
        "3. 不要因为语言流畅就自动高分；要看是否接住深层情绪、是否承接历史、是否专业有边界。\n"
        "4. 若回复直接建议化、说教化、保证化、否定学生感受、忽略风险或编造信息，应扣分。\n"
        "5. 只输出严格 JSON，不要使用 Markdown 代码块。"
    )

    user_prompt = f"""
【评测标准.md】
{rubric_text}

【task_id】
{task.get('task_id')}

【representative_point】
{task.get('representative_point')}

【student_profile_summary】
{inp.get('student_profile_summary')}

【history_until_previous_session】
{inp.get('history_until_previous_session')}

【current_session_event】
{inp.get('current_session_event')}

【current_session_context】
{format_context(inp.get('current_session_context'))}

【current_student_utterance】
Student: {inp.get('current_student_utterance')}

【evaluation_focus】
{json.dumps(task.get('evaluation_focus', {}), ensure_ascii=False, indent=2)}

{reference_block}

【model_response_to_evaluate】
{model_response}

请输出以下 JSON 结构，score 必须是 1-5 的整数：
{{
  "task_id": "{task.get('task_id')}",
  "scores": {{
    "empathy": {{"score": 1, "reason": "..."}},
    "coherence": {{"score": 1, "reason": "..."}},
    "professionalism": {{"score": 1, "reason": "..."}}
  }},
  "overall": {{
    "average_score": 1.0,
    "summary": "..."
  }},
  "risk_flags": ["如无风险写：无"]
}}
""".strip()
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def init_client(cfg: Config) -> Any:
    from openai import OpenAI
    headers: Dict[str, str] = {}
    if cfg.site_url:
        headers["HTTP-Referer"] = cfg.site_url
    if cfg.app_title:
        headers["X-OpenRouter-Title"] = cfg.app_title
    kwargs: Dict[str, Any] = {"api_key": cfg.api_key, "base_url": cfg.base_url, "timeout": cfg.timeout}
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def call_chat(client: Any, *, model: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int, max_retries: int, retry_base_sleep: float) -> str:
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("模型返回内容为空。")
            return content.strip()
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_s = retry_base_sleep * (2 ** attempt) + random.uniform(0, 0.4)
            print(f"[WARN] 调用失败，第 {attempt + 1}/{max_retries + 1} 次重试前等待 {sleep_s:.1f}s：{exc}", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError(f"模型调用失败，已重试 {max_retries} 次：{last_error}")


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(cleaned[start:end + 1])
    raise ValueError(f"无法从评测输出中解析 JSON：{text[:500]}")


def normalize_judgement(task_id: str, raw_text: str) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text)
    parsed.setdefault("task_id", task_id)
    parsed.setdefault("scores", {})
    total = 0
    count = 0
    for metric in METRICS:
        metric_obj = parsed["scores"].setdefault(metric, {})
        if not isinstance(metric_obj, dict):
            metric_obj = {"score": metric_obj, "reason": ""}
            parsed["scores"][metric] = metric_obj
        try:
            score_int = int(round(float(metric_obj.get("score"))))
            score_int = max(1, min(5, score_int))
            metric_obj["score"] = score_int
            total += score_int
            count += 1
        except Exception:
            metric_obj["score"] = None
        metric_obj.setdefault("reason", "")
    parsed.setdefault("overall", {})
    parsed["overall"]["average_score"] = round(total / count, 2) if count else None
    parsed["overall"].setdefault("summary", "")
    parsed.setdefault("risk_flags", ["无"])
    return parsed


def composite_run_id(item: TaskItem) -> str:
    return str(item.task.get("task_id") or f"{item.task_file.stem}#{item.task_index_in_file}")


def load_existing_task_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    ids: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if obj.get("task_id") and not obj.get("error"):
                ids.add(str(obj["task_id"]))
        except json.JSONDecodeError:
            continue
    return ids


def select_task_items(items: Sequence[TaskItem], task_ids: Optional[List[str]], case_files: Optional[List[str]], limit: Optional[int]) -> List[TaskItem]:
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
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if task_id:
            out[task_id] = row
    return out


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def flatten_for_csv(item: Dict[str, Any]) -> Dict[str, Any]:
    judgement = item.get("judgement", {}) or {}
    scores = judgement.get("scores", {}) or {}
    row: Dict[str, Any] = {
        "task_id": item.get("task_id"),
        "case_id": item.get("case_id"),
        "session_id": item.get("session_id"),
        "task_type": item.get("task_type"),
        "representative_point": item.get("representative_point"),
        "target_model": item.get("target_model"),
        "judge_model": item.get("judge_model"),
        "overall_average_score": judgement.get("overall", {}).get("average_score"),
        "overall_summary": judgement.get("overall", {}).get("summary"),
        "risk_flags": "; ".join(map(str, judgement.get("risk_flags", []))),
        "model_response": item.get("model_response"),
        "error": item.get("error"),
    }
    for metric in METRICS:
        row[f"{metric}_score"] = scores.get(metric, {}).get("score")
        row[f"{metric}_reason"] = scores.get(metric, {}).get("reason")
    return row


def rewrite_csv(path: Path, items: List[Dict[str, Any]]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_for_csv(item) for item in items]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: Config) -> None:
    csv_rows = read_csv_rows(cfg.input_csv)
    csv_index = build_csv_index(csv_rows)
    tasks = select_task_items(load_task_items(cfg.tasks), cfg.task_ids, cfg.case_files, cfg.limit)
    rubric_text = sanitize_text(read_text(cfg.rubric))

    if cfg.skip_missing_csv_rows:
        tasks = [item for item in tasks if composite_run_id(item) in csv_index]
    if not tasks:
        raise ValueError("没有匹配到需要评分的任务。")

    if cfg.dry_run:
        first = tasks[0]
        row = csv_index.get(composite_run_id(first), {})
        response = str(row.get("model_response", "")).strip() or "[DRY RUN] CSV 中的 model_response 会放在这里。"
        msgs = build_judge_messages(first.task, response, rubric_text, cfg.include_reference_answer)
        print("=== Dry Run: judge messages ===")
        print(json.dumps(msgs, ensure_ascii=False, indent=2)[:12000])
        print("\n=== Dry Run: sanitized rubric preview ===")
        print(rubric_text[:1500])
        print("\nDry run 完成：没有调用模型，也没有写入结果。")
        return

    client = init_client(cfg)
    existing_ids = load_existing_task_ids(cfg.output) if cfg.resume else set()
    all_items = read_jsonl(cfg.output) if cfg.resume else []
    if not cfg.resume:
        if cfg.output.exists():
            cfg.output.unlink()
        if cfg.csv_output and cfg.csv_output.exists():
            cfg.csv_output.unlink()

    print(f"Loaded {len(csv_rows)} CSV row(s) from: {cfg.input_csv}", flush=True)
    print(f"准备评分 {len(tasks)} 个 SRG 任务；target_model_label={cfg.target_model_label}; judge_model={cfg.judge_model}", flush=True)

    for idx, item in enumerate(tasks, start=1):
        task = item.task
        task_id = str(task.get("task_id", f"{item.task_file.stem}#{item.task_index_in_file}"))
        run_id = composite_run_id(item)
        if run_id in existing_ids:
            print(f"[{idx}/{len(tasks)}] 跳过已完成任务：{run_id}", flush=True)
            continue

        row = csv_index.get(run_id)
        print(f"[{idx}/{len(tasks)}] 评分任务：{item.task_file.name} :: {task_id}", flush=True)
        record: Dict[str, Any] = {
            "run_id": run_id,
            "task_file": str(item.task_file),
            "task_index_in_file": item.task_index_in_file,
            "task_id": task_id,
            "case_id": task.get("case_id"),
            "session_id": task.get("session_id"),
            "task_type": task.get("task_type"),
            "representative_point": task.get("representative_point"),
            "target_model": cfg.target_model_label,
            "judge_model": cfg.judge_model,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": None,
            "model_response": None,
            "judge_raw": None,
            "judgement": None,
            "reference_answer": task.get("reference_answer"),
            "evaluation_focus": task.get("evaluation_focus"),
            "source_csv": str(cfg.input_csv),
            "source_csv_row": row,
            "error": None,
        }
        try:
            if row is None:
                raise ValueError(f"CSV 中找不到 task_id={run_id} 的模型输出。")
            status = str(row.get("status", "")).strip().lower()
            if cfg.skip_non_ok_rows and status and status != "ok":
                raise ValueError(f"CSV 行 status={status!r}，不是 ok。")
            model_response = str(row.get("model_response", "") or "").strip()
            if not model_response:
                raise ValueError("CSV 中 model_response 为空。")
            record["model_response"] = model_response

            judge_messages = build_judge_messages(task, model_response, rubric_text, cfg.include_reference_answer)
            if cfg.print_prompt:
                print("=== Judge Prompt ===")
                print(judge_messages[-1]["content"][:20000])
            raw = call_chat(
                client,
                model=cfg.judge_model,
                messages=judge_messages,
                temperature=cfg.judge_temperature,
                max_tokens=cfg.judge_max_tokens,
                max_retries=cfg.max_retries,
                retry_base_sleep=cfg.retry_base_sleep,
            )
            record["judge_raw"] = raw
            record["judgement"] = normalize_judgement(task_id, raw)
            print(f"    完成：平均分={record['judgement'].get('overall', {}).get('average_score')}", flush=True)
        except Exception as exc:
            record["error"] = str(exc)
            print(f"    [ERROR] {task_id}: {exc}", file=sys.stderr, flush=True)
        finally:
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            append_jsonl(cfg.output, record)
            all_items.append(record)
            if cfg.csv_output:
                rewrite_csv(cfg.csv_output, all_items)
            if cfg.sleep_between_tasks > 0:
                time.sleep(cfg.sleep_between_tasks)

    print(f"全部完成。JSONL: {cfg.output}", flush=True)
    if cfg.csv_output:
        print(f"CSV: {cfg.csv_output}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> Config:
    p = argparse.ArgumentParser(description="OpenRouter + OpenAI SDK：SRG CSV 输出批量评分脚本")
    p.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, type=Path)
    p.add_argument("--tasks", default=DEFAULT_TASKS, type=Path)
    p.add_argument("--rubric", default=DEFAULT_RUBRIC, type=Path)
    p.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    p.add_argument("--csv-output", default=DEFAULT_CSV_OUTPUT, type=Path)
    p.add_argument("--target-model-label", default="local/qwen3-4b-lora", help="仅用于输出记录，不会调用 target model")
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL))
    p.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL))
    p.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    p.add_argument("--site-url", default=os.getenv("OPENROUTER_SITE_URL"))
    p.add_argument("--app-title", default=os.getenv("OPENROUTER_APP_TITLE", "srg-csv-eval-runner"))
    p.add_argument("--judge-temperature", default=0.0, type=float)
    p.add_argument("--judge-max-tokens", default=1200, type=int)
    p.add_argument("--timeout", default=120.0, type=float)
    p.add_argument("--max-retries", default=3, type=int)
    p.add_argument("--retry-base-sleep", default=1.5, type=float)
    p.add_argument("--sleep-between-tasks", default=0.0, type=float)
    p.add_argument("--task-id", action="append", dest="task_ids")
    p.add_argument("--case-file", action="append", dest="case_files")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--print-prompt", action="store_true")
    p.add_argument("--include-reference-answer", action="store_true", help="默认不传 reference_answer；加此参数后作为高分参考方向传给 judge")
    p.add_argument("--no-skip-missing-csv-rows", action="store_true")
    p.add_argument("--include-non-ok-rows", action="store_true")
    args = p.parse_args(argv)

    if not args.dry_run and not args.api_key:
        raise SystemExit("缺少 OPENROUTER_API_KEY。请先 export OPENROUTER_API_KEY='sk-or-v1-...'，或传 --api-key。")
    csv_output = None if args.csv_output is None or str(args.csv_output).strip() == "" else args.csv_output
    return Config(
        input_csv=args.input_csv,
        tasks=args.tasks,
        rubric=args.rubric,
        output=args.output,
        csv_output=csv_output,
        target_model_label=args.target_model_label,
        judge_model=args.judge_model,
        base_url=args.base_url,
        api_key=args.api_key or "",
        site_url=args.site_url,
        app_title=args.app_title,
        judge_temperature=args.judge_temperature,
        judge_max_tokens=args.judge_max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_base_sleep=args.retry_base_sleep,
        sleep_between_tasks=args.sleep_between_tasks,
        task_ids=args.task_ids,
        case_files=args.case_files,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        print_prompt=args.print_prompt,
        include_reference_answer=args.include_reference_answer,
        skip_missing_csv_rows=not args.no_skip_missing_csv_rows,
        skip_non_ok_rows=not args.include_non_ok_rows,
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()

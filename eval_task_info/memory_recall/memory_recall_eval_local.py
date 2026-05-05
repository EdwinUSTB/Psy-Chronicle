

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
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRICS = ("accuracy", "completeness", "temporal_consistency", "no_hallucination")

SECRET_PATTERNS = [
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    re.compile(r"(?im)^\s*psy\s*api\s*[:：].*$"),
    re.compile(r"(?im)^\s*(api\s*key|openrouter_api_key|OPENROUTER_API_KEY)\s*[:=：].*$"),
]


@dataclass
class RunConfig:
    input_csv_path: Path
    rubric_path: Path
    output_path: Path
    csv_output_path: Optional[Path]
    full_session_dir: Optional[Path]
    tasks_path: Optional[Path]
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
    limit: Optional[int]
    resume: bool
    dry_run: bool
    print_prompt: bool
    include_reference_answer: bool
    include_answer_source: bool
    max_full_session_chars: Optional[int]
    truncate_side: str


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def sanitize_text(text: str) -> str:
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def iter_json_files(path: Path) -> Iterable[Path]:
    if path.is_file() and path.suffix.lower() == ".json":
        yield path
        return
    if path.is_dir():
        for p in sorted(path.rglob("*.json")):
            if p.name.lower().endswith("_fullsession.json"):
                continue
            yield p


def normalize_task_list(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if "task_id" in data:
            return [data]
        for key in ("tasks", "data", "items"):
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
    return []


def load_task_index(tasks_path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """Optional: load original MR task files for question/evaluation_focus/reference if CSV lacks them."""
    if not tasks_path:
        return {}
    if not tasks_path.exists():
        raise FileNotFoundError(f"tasks path not found: {tasks_path}")

    task_index: Dict[str, Dict[str, Any]] = {}
    for fp in iter_json_files(tasks_path):
        try:
            data = read_json(fp)
        except Exception as exc:
            print(f"[WARN] 跳过无法读取的任务文件 {fp}: {exc}", file=sys.stderr)
            continue
        for task in normalize_task_list(data):
            task_id = str(task.get("task_id", "")).strip()
            if task_id:
                task_index[task_id] = task
    return task_index


def clean_case_stem(stem: str) -> str:
    return stem.replace("_", "").replace(" ", "")


def candidate_full_session_files(row: Dict[str, Any], full_session_dir: Optional[Path]) -> List[Path]:
    candidates: List[Path] = []

    # 1. 优先用本地推理 CSV 自带的 full_session_file
    fs = str(row.get("full_session_file", "") or "").strip()
    if fs:
        candidates.append(Path(fs))

    if not full_session_dir:
        return candidates

    stems: List[str] = []

    source_file = str(row.get("source_file", "") or "").strip()
    if source_file:
        stems.append(Path(source_file).stem)

    case_id = str(row.get("case_id", "") or "").strip()
    if case_id:
        stems.append(case_id)
        stems.append(clean_case_stem(case_id))

    # fallback，不一定能从 task_id 还原中文名，但保留候选
    task_id = str(row.get("task_id", "") or "").strip()
    if task_id:
        stems.append(task_id)

    seen = set()
    for stem in stems:
        for s in (stem, clean_case_stem(stem)):
            if s and s not in seen:
                seen.add(s)
                candidates.append(full_session_dir / f"{s}_fullsession.json")

    return candidates


def find_full_session_file(row: Dict[str, Any], full_session_dir: Optional[Path]) -> Optional[Path]:
    for p in candidate_full_session_files(row, full_session_dir):
        if p.exists():
            return p
    return None


def format_full_session(data: Any) -> str:
    if isinstance(data, list):
        lines: List[str] = []
        for idx, turn in enumerate(data, 1):
            if isinstance(turn, dict):
                role = str(turn.get("role", "")).strip()
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue
                if role.lower() == "student":
                    role_zh = "学生"
                elif role.lower() == "counselor":
                    role_zh = "咨询师"
                else:
                    role_zh = role or "未知角色"
                lines.append(f"{idx}. {role_zh}：{content}")
            else:
                lines.append(f"{idx}. {turn}")
        return "\n".join(lines)
    if isinstance(data, dict):
        return json.dumps(data, ensure_ascii=False, indent=2)
    return str(data)


def load_full_session_text(path: Optional[Path], max_chars: Optional[int], truncate_side: str) -> str:
    if not path:
        return ""
    data = read_json(path)
    text = format_full_session(data)

    if max_chars is not None and len(text) > max_chars:
        if truncate_side == "head":
            return f"【完整历史过长，仅保留开头 {max_chars} 字符】\n" + text[:max_chars]
        return f"【完整历史过长，仅保留末尾 {max_chars} 字符】\n" + text[-max_chars:]

    return text


def build_judge_messages(
    row: Dict[str, Any],
    task: Optional[Dict[str, Any]],
    model_response: str,
    rubric_text: str,
    full_session_text: str,
    include_reference_answer: bool,
    include_answer_source: bool,
) -> List[Dict[str, str]]:
    task = task or {}

    task_id = str(row.get("task_id", "") or task.get("task_id", "")).strip()
    case_id = str(row.get("case_id", "") or task.get("case_id", "")).strip()
    task_type = str(row.get("task_type", "") or task.get("task_type", "")).strip()
    question = str(row.get("question", "") or task.get("question", "")).strip()
    evaluation_focus = task.get("evaluation_focus", {})

    optional_blocks: List[str] = []
    if evaluation_focus:
        optional_blocks.append(
            "【evaluation_focus，若有则作为该题额外关注点】\n"
            + json.dumps(evaluation_focus, ensure_ascii=False, indent=2)
        )

    if include_reference_answer and task.get("reference_answer"):
        optional_blocks.append(
            "【reference_answer，可作为辅助参考；不要要求逐字相似】\n"
            + str(task.get("reference_answer"))
        )

    if include_answer_source and task.get("answer_source"):
        optional_blocks.append(
            "【answer_source，可作为辅助定位；不要替代完整历史核验】\n"
            + json.dumps(task.get("answer_source"), ensure_ascii=False, indent=2)
        )

    optional_text = "\n\n".join(optional_blocks)

    system_prompt = (
        "你是严格、审慎的 MR(memory recall) 任务评测员。"
        "请基于评分标准、完整咨询历史、问题和模型回答，对模型回答进行评分。\n\n"
        "评分规则：\n"
        "1. 分别给 accuracy、completeness、temporal_consistency、no_hallucination 四项打 0-5 的整数分。\n"
        "2. 必须优先根据完整咨询历史核验事实，不要根据常识或猜测补全。\n"
        "3. 不要因为语言流畅就自动高分；重点看事实是否准确、要点是否覆盖、时间顺序是否正确、是否编造。\n"
        "4. 若回答为空、答非所问、明显编造，应给低分。\n"
        "5. 只输出严格 JSON，不要使用 Markdown 代码块。"
    )

    user_prompt = f"""
【评分标准.md】
{rubric_text}

【task_id】
{task_id}

【case_id】
{case_id}

【task_type】
{task_type}

【question】
{question}

【完整咨询历史 full_session】
{full_session_text}

{optional_text}

【model_response_to_evaluate】
{model_response}

请输出以下 JSON 结构，score 必须是 0-5 的整数：
{{
  "task_id": "{task_id}",
  "scores": {{
    "accuracy": {{"score": 0, "reason": "..."}},
    "completeness": {{"score": 0, "reason": "..."}},
    "temporal_consistency": {{"score": 0, "reason": "..."}},
    "no_hallucination": {{"score": 0, "reason": "..."}}
  }},
  "overall": {{
    "average_score": 0.0,
    "summary": "..."
  }},
  "risk_flags": ["如无风险写：无"]
}}
""".strip()

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def init_client(cfg: RunConfig) -> Any:
    from openai import OpenAI

    headers: Dict[str, str] = {}
    if cfg.site_url:
        headers["HTTP-Referer"] = cfg.site_url
    if cfg.app_title:
        headers["X-OpenRouter-Title"] = cfg.app_title

    kwargs: Dict[str, Any] = {
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
        "timeout": cfg.timeout,
    }
    if headers:
        kwargs["default_headers"] = headers
    return OpenAI(**kwargs)


def call_chat(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_base_sleep: float,
) -> str:
    last_error: Optional[BaseException] = None

    for attempt in range(max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            if not content:
                raise RuntimeError("模型返回内容为空。")
            return content.strip()
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_s = retry_base_sleep * (2**attempt) + random.uniform(0, 0.4)
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
        return json.loads(cleaned[start : end + 1])

    raise ValueError(f"无法从评测输出中解析 JSON：{text[:500]}")


def normalize_judgement(task_id: str, raw_text: str) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text)
    parsed.setdefault("task_id", task_id)
    parsed.setdefault("scores", {})

    total = 0
    count = 0

    for metric in METRICS:
        metric_obj = parsed["scores"].setdefault(metric, {})
        # 兼容 {"accuracy": 4} 这种错误格式
        if not isinstance(metric_obj, dict):
            metric_obj = {"score": metric_obj, "reason": ""}
            parsed["scores"][metric] = metric_obj

        score = metric_obj.get("score")
        try:
            score_int = int(round(float(score)))
            score_int = max(0, min(5, score_int))
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


def load_existing_task_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    task_ids: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if item.get("task_id") and not item.get("error"):
                task_ids.add(str(item["task_id"]))
        except json.JSONDecodeError:
            continue
    return task_ids


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return items


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def flatten_for_csv(item: Dict[str, Any]) -> Dict[str, Any]:
    judgement = item.get("judgement", {}) or {}
    scores = judgement.get("scores", {}) or {}

    row: Dict[str, Any] = {
        "task_id": item.get("task_id"),
        "case_id": item.get("case_id"),
        "task_type": item.get("task_type"),
        "source_file": item.get("source_file"),
        "full_session_file": item.get("full_session_file"),
        "judge_model": item.get("judge_model"),
        "overall_average_score": judgement.get("overall", {}).get("average_score"),
        "overall_summary": judgement.get("overall", {}).get("summary"),
        "risk_flags": "; ".join(map(str, judgement.get("risk_flags", []))),
        "question": item.get("question"),
        "model_response": item.get("model_response"),
        "error": item.get("error"),
    }

    for metric in METRICS:
        metric_obj = scores.get(metric, {}) or {}
        row[f"{metric}_score"] = metric_obj.get("score")
        row[f"{metric}_reason"] = metric_obj.get("reason")

    return row


def rewrite_csv(path: Path, items: List[Dict[str, Any]]) -> None:
    if not items:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_for_csv(item) for item in items]
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_item_from_row(row: Dict[str, Any], cfg: RunConfig, task: Optional[Dict[str, Any]], full_session_file: Optional[Path]) -> Dict[str, Any]:
    task = task or {}
    return {
        "task_id": str(row.get("task_id", "") or task.get("task_id", "")).strip(),
        "case_id": str(row.get("case_id", "") or task.get("case_id", "")).strip(),
        "task_type": str(row.get("task_type", "") or task.get("task_type", "")).strip(),
        "source_file": str(row.get("source_file", "")).strip(),
        "full_session_file": str(full_session_file) if full_session_file else "",
        "judge_model": cfg.judge_model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "question": str(row.get("question", "") or task.get("question", "")).strip(),
        "model_response": str(row.get("model_response", "")).strip(),
        "judge_raw": None,
        "judgement": None,
        "evaluation_focus": task.get("evaluation_focus"),
        "error": None,
    }


def run(cfg: RunConfig) -> None:
    rows = read_csv_rows(cfg.input_csv_path)
    rubric_text = sanitize_text(read_text(cfg.rubric_path))
    task_index = load_task_index(cfg.tasks_path)

    if cfg.task_ids:
        wanted = set(cfg.task_ids)
        rows = [row for row in rows if str(row.get("task_id")) in wanted]
    if cfg.limit is not None:
        rows = rows[: cfg.limit]

    if not rows:
        raise ValueError("没有匹配到需要评分的 CSV 行。")

    if cfg.dry_run:
        first = rows[0]
        task = task_index.get(str(first.get("task_id", "")).strip())
        full_session_file = find_full_session_file(first, cfg.full_session_dir)
        full_session_text = load_full_session_text(full_session_file, cfg.max_full_session_chars, cfg.truncate_side) if full_session_file else ""
        messages = build_judge_messages(
            first,
            task,
            str(first.get("model_response", "")).strip(),
            rubric_text,
            full_session_text,
            cfg.include_reference_answer,
            cfg.include_answer_source,
        )
        print("=== Dry Run: judge messages ===")
        print(json.dumps(messages, ensure_ascii=False, indent=2)[:12000])
        print("\nDry run 完成：没有调用模型，也没有写入结果。")
        return

    client = init_client(cfg)
    existing_ids = load_existing_task_ids(cfg.output_path) if cfg.resume else set()
    all_items = read_jsonl(cfg.output_path) if cfg.resume else []

    print(f"准备评分 {len(rows)} 行；judge_model={cfg.judge_model}")
    if cfg.tasks_path:
        print(f"已加载原始任务索引 {len(task_index)} 条：{cfg.tasks_path}")
    if existing_ids:
        print(f"resume=true，已存在 {len(existing_ids)} 个 task_id，将自动跳过。")

    for idx, row in enumerate(rows, start=1):
        task_id = str(row.get("task_id", "")).strip()

        if task_id in existing_ids:
            print(f"[{idx}/{len(rows)}] 跳过已完成任务：{task_id}")
            continue

        task = task_index.get(task_id)
        full_session_file = find_full_session_file(row, cfg.full_session_dir)

        item = make_item_from_row(row, cfg, task, full_session_file)

        print(f"[{idx}/{len(rows)}] 评分任务：{task_id}")

        try:
            if not item["model_response"]:
                raise ValueError("CSV 中 model_response 为空。")
            if not item["question"]:
                raise ValueError("缺少 question。请确认 CSV 或 --tasks 中包含 question。")
            if not full_session_file:
                raise FileNotFoundError("未找到对应 full_session 文件；请检查 CSV 的 full_session_file 或 --full-session-dir。")

            full_session_text = load_full_session_text(
                full_session_file,
                cfg.max_full_session_chars,
                cfg.truncate_side,
            )

            judge_messages = build_judge_messages(
                row,
                task,
                item["model_response"],
                rubric_text,
                full_session_text,
                cfg.include_reference_answer,
                cfg.include_answer_source,
            )

            if cfg.print_prompt:
                print("=== Judge Prompt ===")
                print(judge_messages[-1]["content"][:20000])

            judge_raw = call_chat(
                client,
                model=cfg.judge_model,
                messages=judge_messages,
                temperature=cfg.judge_temperature,
                max_tokens=cfg.judge_max_tokens,
                max_retries=cfg.max_retries,
                retry_base_sleep=cfg.retry_base_sleep,
            )

            item["judge_raw"] = judge_raw
            item["judgement"] = normalize_judgement(task_id, judge_raw)
            avg = item["judgement"].get("overall", {}).get("average_score")
            print(f"    完成：平均分={avg}")

        except Exception as exc:
            item["error"] = str(exc)
            print(f"    [ERROR] {task_id}: {exc}", file=sys.stderr)

        finally:
            item["finished_at"] = datetime.now(timezone.utc).isoformat()
            append_jsonl(cfg.output_path, item)
            all_items.append(item)
            if cfg.csv_output_path:
                rewrite_csv(cfg.csv_output_path, all_items)
            if cfg.sleep_between_tasks > 0:
                time.sleep(cfg.sleep_between_tasks)

    print(f"全部完成。JSONL: {cfg.output_path}")
    if cfg.csv_output_path:
        print(f"CSV: {cfg.csv_output_path}")


def parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    parser = argparse.ArgumentParser(description="OpenRouter + OpenAI SDK：MR CSV 输出批量评分脚本")
    parser.add_argument("--input-csv", required=True, type=Path, help="本地模型 MR 输出 CSV，例如 mr_qwen3_4b_lora.csv")
    parser.add_argument("--rubric", required=True, type=Path, help="MR 评分标准 markdown 文件路径")
    parser.add_argument("--output", default=Path("mr_graded.jsonl"), type=Path, help="输出 JSONL 路径")
    parser.add_argument("--csv-output", default=Path("mr_graded.csv"), type=Path, help="输出 CSV 路径；传空字符串可禁用")
    parser.add_argument("--full-session-dir", default=None, type=Path, help="full_seesion/full_session 文件夹；CSV 有 full_session_file 时也建议传入作为 fallback")
    parser.add_argument("--tasks", default=None, type=Path, help="可选：MR 原始任务 JSON 文件或文件夹，用于补充 evaluation_focus/question；默认不需要")

    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "openai/gpt-5.2"), help="评测模型名称")
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), help="OpenRouter base_url")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API Key；建议用环境变量 OPENROUTER_API_KEY")
    parser.add_argument("--site-url", default=os.getenv("OPENROUTER_SITE_URL"), help="可选 HTTP-Referer")
    parser.add_argument("--app-title", default=os.getenv("OPENROUTER_APP_TITLE", "mr-csv-eval-runner"), help="可选 X-OpenRouter-Title")
    parser.add_argument("--judge-temperature", default=0.0, type=float)
    parser.add_argument("--judge-max-tokens", default=1600, type=int)
    parser.add_argument("--timeout", default=180.0, type=float)
    parser.add_argument("--max-retries", default=3, type=int)
    parser.add_argument("--retry-base-sleep", default=1.5, type=float)
    parser.add_argument("--sleep-between-tasks", default=0.0, type=float, help="任务间隔，避免触发限流")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="只评分指定 task_id；可重复传入")
    parser.add_argument("--limit", type=int, default=None, help="只评分前 N 行")
    parser.add_argument("--resume", action="store_true", help="若 output 已有同名 task_id 且无 error，则跳过")
    parser.add_argument("--dry-run", action="store_true", help="打印首行评测提示词，不调用模型")
    parser.add_argument("--print-prompt", action="store_true", help="逐条打印发送给 judge 的 prompt 片段")

    parser.add_argument("--max-full-session-chars", default=None, type=int, help="限制 full_session 字符数；默认不截断")
    parser.add_argument("--truncate-side", choices=("head", "tail"), default="tail", help="超长截断时保留开头还是末尾；默认 tail")

    parser.add_argument("--include-reference-answer", action="store_true", help="默认不传；传入后会从 --tasks 中读取 reference_answer 作为辅助参考")
    parser.add_argument("--include-answer-source", action="store_true", help="默认不传；传入后会从 --tasks 中读取 answer_source 作为辅助定位")

    args = parser.parse_args(argv)

    if not args.dry_run and not args.api_key:
        raise SystemExit("缺少 OPENROUTER_API_KEY。请先 export OPENROUTER_API_KEY='sk-or-v1-...'，或传 --api-key。")

    csv_output_path: Optional[Path]
    if args.csv_output is None or str(args.csv_output).strip() == "":
        csv_output_path = None
    else:
        csv_output_path = args.csv_output

    return RunConfig(
        input_csv_path=args.input_csv,
        rubric_path=args.rubric,
        output_path=args.output,
        csv_output_path=csv_output_path,
        full_session_dir=args.full_session_dir,
        tasks_path=args.tasks,
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
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        print_prompt=args.print_prompt,
        include_reference_answer=args.include_reference_answer,
        include_answer_source=args.include_answer_source,
        max_full_session_chars=args.max_full_session_chars,
        truncate_side=args.truncate_side,
    )


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()

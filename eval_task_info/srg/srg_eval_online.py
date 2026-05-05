
"""
OpenRouter + OpenAI SDK 批量测试脚本

在eval_runner.py的基础上，增加了一个功能：可以一次性评测一个目录下的所有任务文件，并将结果保存到一个总的jsonl和csv文件中。

conda activate psy
export OPENROUTER_API_KEY="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"

python eval_runner_all.py \
  --tasks . \
  --rubric 评测标准.md \
  --target-model qwen/qwen3-8b \
  --judge-model openai/gpt-5.2 \
  --output all_results_qwen3-8b.jsonl \
  --csv-output all_results_qwen3-8b.csv

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
from typing import Any, Dict, Iterable, List, Optional, Tuple


METRICS = ("empathy", "coherence", "professionalism")

SECRET_PATTERNS = [
    # OpenRouter key pattern. Kept broad enough to catch accidental variants.
    re.compile(r"sk-or-v1-[A-Za-z0-9_-]+"),
    # Lines like: psy api: sk-...
    re.compile(r"(?im)^\s*psy\s*api\s*[:：].*$"),
    # Lines like: api key: sk-...
    re.compile(r"(?im)^\s*(api\s*key|openrouter_api_key|OPENROUTER_API_KEY)\s*[:=：].*$"),
]


@dataclass
class RunConfig:
    tasks_path: Path
    rubric_path: Path
    output_path: Path
    csv_output_path: Optional[Path]
    target_model: str
    judge_model: str
    base_url: str
    api_key: str
    site_url: Optional[str]
    app_title: Optional[str]
    generation_temperature: float
    judge_temperature: float
    generation_max_tokens: int
    judge_max_tokens: int
    timeout: float
    max_retries: int
    retry_base_sleep: float
    sleep_between_tasks: float
    task_ids: Optional[List[str]]
    limit: Optional[int]
    resume: bool
    dry_run: bool
    include_reference_answer: bool


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")


def sanitize_text(text: str) -> str:
    """Remove accidental secrets before sending rubric/content to any model."""
    sanitized = text
    for pattern in SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


def _load_single_task_file(path: Path) -> List[Dict[str, Any]]:
    raw_text = read_text(path)
    
    # 1. 增加非空校验，跳过空白文件
    if not raw_text.strip():
        print(f"[警告] 跳过空文件: {path}")
        return []

    # 2. 增加异常捕获，明确指出是哪个文件格式不对
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"解析出错！文件 {path} 不是合法的 JSON 格式，详细错误: {e}")

    if isinstance(data, list):
        tasks = data
    elif isinstance(data, dict):
        # 常见包装格式兼容：{"tasks": [...]}, {"data": [...]}, {"items": [...]}
        for key in ("tasks", "data", "items"):
            if isinstance(data.get(key), list):
                tasks = data[key]
                break
        else:
            raise ValueError(f"文件 {path} 任务 JSON 必须是 list，或包含 tasks/data/items 列表字段。")
    else:
        raise ValueError(f"文件 {path} 任务 JSON 顶层必须是 list 或 dict。")

    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"文件 {path} 第 {idx} 个任务不是对象。")
        if "task_id" not in task:
            raise ValueError(f"文件 {path} 第 {idx} 个任务缺少 task_id。")
        if "input_to_model" not in task or not isinstance(task["input_to_model"], dict):
            raise ValueError(f"文件 {path} 任务 {task.get('task_id', idx)} 缺少 input_to_model 对象。")
        if "evaluation_focus" not in task or not isinstance(task["evaluation_focus"], dict):
            raise ValueError(f"文件 {path} 任务 {task.get('task_id', idx)} 缺少 evaluation_focus 对象。")
    return tasks


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    if path.is_dir():
        all_tasks = []
        # 按名称排序以保持确定的执行顺序
        for json_file in sorted(path.glob("*.json")):
            all_tasks.extend(_load_single_task_file(json_file))
        return all_tasks
    else:
        return _load_single_task_file(path)


def format_context(context: Any) -> str:
    if not context:
        return "无"
    if not isinstance(context, list):
        return str(context)

    lines: List[str] = []
    for turn in context:
        if isinstance(turn, dict):
            role = turn.get("role", "Unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(turn))
    return "\n".join(lines)


def build_generation_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    inp = task["input_to_model"]
    required_fields = [
        "student_profile_summary",
        "history_until_previous_session",
        "current_session_event",
        "current_session_context",
        "current_student_utterance",
    ]
    missing = [field for field in required_fields if field not in inp]
    if missing:
        raise ValueError(f"任务 {task['task_id']} 的 input_to_model 缺少字段: {', '.join(missing)}")

    system_prompt = (
        "你是一名受过专业训练的高校心理咨询师。你的任务是根据给定学生画像、历史会话、"
        "当前事件、当前会话上下文和学生最新发言，生成咨询师下一轮回应。\n\n"
        "要求：\n"
        "1. 使用中文，语气温和、稳健、真诚。\n"
        "2. 优先深层共情，准确命名学生的情绪、冲突和意义；不要急于给建议。\n"
        "3. 紧密承接历史轨迹和当前 session，不要只回应表层句子。\n"
        "4. 保持校园心理咨询专业边界和风险敏感性；避免保证、评判、说教、直接替学生做决定。\n"
        "5. 输出咨询师回复本身，不要输出分析、标题、评分或 JSON。\n"
        "6. 建议 1-2 段，最后可以用一个开放式探索问题收束。"
    )

    user_prompt = f"""
请生成 Counselor 对 Student 的下一轮回复。

【task_id】
{task.get('task_id')}

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
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_judge_messages(
    task: Dict[str, Any],
    model_response: str,
    rubric_text: str,
    include_reference_answer: bool = False,
) -> List[Dict[str, str]]:
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

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def init_client(cfg: RunConfig) -> Any:
    from openai import OpenAI  # 延迟导入，方便 dry-run/py_compile 不依赖已安装 SDK

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
        except Exception as exc:  # noqa: BLE001 - CLI runner needs broad retry handling
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_s = retry_base_sleep * (2**attempt) + random.uniform(0, 0.4)
            print(f"[WARN] 调用失败，第 {attempt + 1}/{max_retries + 1} 次重试前等待 {sleep_s:.1f}s：{exc}", file=sys.stderr)
            time.sleep(sleep_s)
    raise RuntimeError(f"模型调用失败，已重试 {max_retries} 次：{last_error}")


def extract_json_object(text: str) -> Dict[str, Any]:
    """Parse judge JSON. Handles accidental ```json fences or leading/trailing prose."""
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
        candidate = cleaned[start : end + 1]
        return json.loads(candidate)
    raise ValueError(f"无法从评测输出中解析 JSON：{text[:500]}")


def normalize_judgement(task_id: str, raw_text: str) -> Dict[str, Any]:
    parsed = extract_json_object(raw_text)
    parsed.setdefault("task_id", task_id)
    parsed.setdefault("scores", {})

    total = 0
    count = 0
    for metric in METRICS:
        metric_obj = parsed["scores"].setdefault(metric, {})
        score = metric_obj.get("score")
        try:
            score_int = int(score)
            score_int = max(1, min(5, score_int))
            metric_obj["score"] = score_int
            total += score_int
            count += 1
        except Exception:
            metric_obj["score"] = None
        metric_obj.setdefault("reason", "")

    parsed.setdefault("overall", {})
    if count:
        parsed["overall"]["average_score"] = round(total / count, 2)
    else:
        parsed["overall"].setdefault("average_score", None)
    parsed["overall"].setdefault("summary", "")
    parsed.setdefault("risk_flags", ["无"])
    return parsed


def load_existing_task_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    task_ids = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            if item.get("task_id"):
                task_ids.add(str(item["task_id"]))
        except json.JSONDecodeError:
            continue
    return task_ids


def append_jsonl(path: Path, item: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def flatten_for_csv(item: Dict[str, Any]) -> Dict[str, Any]:
    judgement = item.get("judgement", {})
    scores = judgement.get("scores", {})
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
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def run(cfg: RunConfig) -> None:
    tasks = load_tasks(cfg.tasks_path)
    rubric_text = sanitize_text(read_text(cfg.rubric_path))

    if cfg.task_ids:
        wanted = set(cfg.task_ids)
        tasks = [task for task in tasks if str(task.get("task_id")) in wanted]
    if cfg.limit is not None:
        tasks = tasks[: cfg.limit]

    if not tasks:
        raise ValueError("没有匹配到需要运行的任务。")

    if cfg.dry_run:
        first = tasks[0]
        print("=== Dry Run: generation messages ===")
        print(json.dumps(build_generation_messages(first), ensure_ascii=False, indent=2))
        print("\n=== Dry Run: sanitized rubric preview ===")
        print(rubric_text[:1500])
        print("\nDry run 完成：没有调用模型，也没有写入结果。")
        return

    client = init_client(cfg)
    existing_ids = load_existing_task_ids(cfg.output_path) if cfg.resume else set()
    all_items = read_jsonl(cfg.output_path) if cfg.resume else []

    print(f"准备运行 {len(tasks)} 个任务；target_model={cfg.target_model}; judge_model={cfg.judge_model}")
    if existing_ids:
        print(f"resume=true，已存在 {len(existing_ids)} 个 task_id，将自动跳过。")

    for idx, task in enumerate(tasks, start=1):
        task_id = str(task["task_id"])
        if task_id in existing_ids:
            print(f"[{idx}/{len(tasks)}] 跳过已完成任务：{task_id}")
            continue

        print(f"[{idx}/{len(tasks)}] 运行任务：{task_id}")
        started_at = datetime.now(timezone.utc).isoformat()
        item: Dict[str, Any] = {
            "task_id": task_id,
            "case_id": task.get("case_id"),
            "session_id": task.get("session_id"),
            "task_type": task.get("task_type"),
            "representative_point": task.get("representative_point"),
            "target_model": cfg.target_model,
            "judge_model": cfg.judge_model,
            "started_at": started_at,
            "finished_at": None,
            "model_response": None,
            "judge_raw": None,
            "judgement": None,
            "reference_answer": task.get("reference_answer"),
            "evaluation_focus": task.get("evaluation_focus"),
            "error": None,
        }

        try:
            generation_messages = build_generation_messages(task)
            model_response = call_chat(
                client,
                model=cfg.target_model,
                messages=generation_messages,
                temperature=cfg.generation_temperature,
                max_tokens=cfg.generation_max_tokens,
                max_retries=cfg.max_retries,
                retry_base_sleep=cfg.retry_base_sleep,
            )
            item["model_response"] = model_response

            judge_messages = build_judge_messages(
                task,
                model_response,
                rubric_text,
                include_reference_answer=cfg.include_reference_answer,
            )
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
        except Exception as exc:  # noqa: BLE001
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
    parser = argparse.ArgumentParser(description="OpenRouter + OpenAI SDK 批量生成与评测脚本")
    parser.add_argument("--tasks", required=True, type=Path, help="任务 JSON 文件路径，或包含 JSON 的文件夹路径")
    parser.add_argument("--rubric", required=True, type=Path, help="评测标准 markdown 文件路径")
    parser.add_argument("--output", default=Path("results.jsonl"), type=Path, help="输出 JSONL 路径")
    parser.add_argument("--csv-output", default=Path("results.csv"), type=Path, help="输出 CSV 路径；传空字符串可禁用")
    parser.add_argument("--target-model", default=os.getenv("TARGET_MODEL", "openai/gpt-5.2"), help="被测模型名称")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", os.getenv("TARGET_MODEL", "openai/gpt-5.2")), help="评测模型名称")
    parser.add_argument("--base-url", default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"), help="OpenRouter base_url")
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"), help="OpenRouter API Key；建议用环境变量 OPENROUTER_API_KEY")
    parser.add_argument("--site-url", default=os.getenv("OPENROUTER_SITE_URL"), help="可选 HTTP-Referer")
    parser.add_argument("--app-title", default=os.getenv("OPENROUTER_APP_TITLE", "counseling-eval-runner"), help="可选 X-OpenRouter-Title")
    parser.add_argument("--generation-temperature", default=0.4, type=float)
    parser.add_argument("--judge-temperature", default=0.0, type=float)
    parser.add_argument("--generation-max-tokens", default=800, type=int)
    parser.add_argument("--judge-max-tokens", default=1200, type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--max-retries", default=3, type=int)
    parser.add_argument("--retry-base-sleep", default=1.5, type=float)
    parser.add_argument("--sleep-between-tasks", default=0.0, type=float, help="任务间隔，避免触发限流")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="只运行指定 task_id；可重复传入")
    parser.add_argument("--limit", type=int, default=None, help="只运行前 N 个任务")
    parser.add_argument("--resume", action="store_true", help="若 output 已有同名 task_id，则跳过")
    parser.add_argument("--dry-run", action="store_true", help="打印首个任务提示词和脱敏后的 rubric，不调用模型")
    parser.add_argument("--include-reference-answer", action="store_true", help="评测时把 reference_answer 作为高分参考方向传给 judge；默认不传")

    args = parser.parse_args(argv)

    if not args.dry_run and not args.api_key:
        raise SystemExit("缺少 OPENROUTER_API_KEY。请先 export OPENROUTER_API_KEY='sk-or-v1-...'，或传 --api-key。")

    csv_output_path: Optional[Path]
    if args.csv_output is None or str(args.csv_output).strip() == "":
        csv_output_path = None
    else:
        csv_output_path = args.csv_output

    return RunConfig(
        tasks_path=args.tasks,
        rubric_path=args.rubric,
        output_path=args.output,
        csv_output_path=csv_output_path,
        target_model=args.target_model,
        judge_model=args.judge_model,
        base_url=args.base_url,
        api_key=args.api_key or "",
        site_url=args.site_url,
        app_title=args.app_title,
        generation_temperature=args.generation_temperature,
        judge_temperature=args.judge_temperature,
        generation_max_tokens=args.generation_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        timeout=args.timeout,
        max_retries=args.max_retries,
        retry_base_sleep=args.retry_base_sleep,
        sleep_between_tasks=args.sleep_between_tasks,
        task_ids=args.task_ids,
        limit=args.limit,
        resume=args.resume,
        dry_run=args.dry_run,
        include_reference_answer=args.include_reference_answer,
    )


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()

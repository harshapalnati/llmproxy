#!/usr/bin/env python
"""
Lightweight tool-call benchmark to validate proxy vs direct behavior.

This is not the full ToolBench dataset; it's a minimal harness with a handful of
tool-use prompts to measure whether the model produces correct tool calls
(name + required args). Extend `DEFAULT_TASKS` or provide a JSONL via --tasks.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests


Mode = str  # proxy_on | proxy_off | direct


# Tool schema mirrors earlier mock tools.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a user",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["recipient", "body"],
            },
        },
    },
]


@dataclass
class Task:
    prompt: str
    expected_tool: str
    required_args: List[str]


DEFAULT_TASKS = [
    Task(
        prompt="What is the weather in Tokyo today? Return in celsius.",
        expected_tool="get_weather",
        required_args=["location"],
    ),
    Task(
        prompt="Send an email to boss@company.com saying the report is done.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Check the weather in New York and respond in Fahrenheit.",
        expected_tool="get_weather",
        required_args=["location"],
    ),
    Task(
        prompt="Email alice@example.com to confirm the 3pm meeting.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Is it raining in Paris? Use tools if needed.",
        expected_tool="get_weather",
        required_args=["location"],
    ),
]


def load_tasks(path: Optional[str]) -> List[Task]:
    if not path:
        return DEFAULT_TASKS
    tasks: List[Task] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            tasks.append(
                Task(
                    prompt=obj["prompt"],
                    expected_tool=obj["expected_tool"],
                    required_args=obj.get("required_args", []),
                )
            )
    return tasks


def build_url_and_headers(mode: Mode) -> (str, Dict[str, str]):
    proxy_url = os.getenv("PROXY_URL", "http://localhost:9000/v1")
    positron_url = os.getenv("POSITRON_URL", "https://api.positron.ai/v1")
    api_key = os.getenv("POSITRON_KEY")
    if not api_key:
        raise RuntimeError("Missing POSITRON_KEY")

    if mode in ("proxy_on", "proxy_off"):
        url = f"{proxy_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        if mode == "proxy_off":
            headers["x-raph-mode"] = "off"
    else:
        url = f"{positron_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}

    return url, headers


def evaluate_task(task: Task, mode: Mode, model: str, temperature: float) -> bool:
    url, headers = build_url_and_headers(mode)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a function calling assistant. Call the right tool with JSON arguments.",
            },
            {"role": "user", "content": task.prompt},
        ],
        "tools": TOOLS,
        "temperature": temperature,
    }

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        print(f"Request failed ({resp.status_code}): {resp.text}")
        return False

    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return False

    message = choices[0].get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return False

    call = tool_calls[0]
    fn = call.get("function") or {}
    name = fn.get("name")
    args_str = fn.get("arguments") or "{}"
    try:
        args = json.loads(args_str)
    except Exception:
        return False

    if name != task.expected_tool:
        return False
    for req in task.required_args:
        if req not in args:
            return False
    return True


def run(tasks: List[Task], mode: Mode, model: str, temperature: float, limit: int) -> float:
    total = min(limit, len(tasks)) if limit else len(tasks)
    success = 0
    for i, task in enumerate(tasks):
        if i >= total:
            break
        ok = evaluate_task(task, mode, model, temperature)
        if ok:
            success += 1
        print(f"Task {i+1}/{total}: {'✅' if ok else '❌'} ({task.prompt[:60]}...)")
    acc = success / total * 100 if total else 0
    print(f"\nTool-call success ({mode}): {acc:.2f}% ({success}/{total})")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal tool-call benchmark (proxy vs direct).")
    parser.add_argument("--mode", choices=["proxy_on", "proxy_off", "direct"], default="proxy_on")
    parser.add_argument("--model", default=os.getenv("MMLU_MODEL", "llama-3.1-8b-instruct-good-tp2"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tasks", help="Path to JSONL with fields: prompt, expected_tool, required_args")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tasks to run (0 = all)")
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    if not tasks:
        print("No tasks to run.")
        sys.exit(1)

    try:
        run(tasks, mode=args.mode, model=args.model, temperature=args.temperature, limit=args.limit)
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

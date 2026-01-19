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
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search internal documentation",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_flight",
            "description": "Book a flight for a traveler",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Schedule a meeting with participants",
            "parameters": {
                "type": "object",
                "properties": {
                    "participants": {"type": "array", "items": {"type": "string"}},
                    "time": {"type": "string"},
                },
                "required": ["participants", "time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the latest stock price",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text into a target language",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target_language": {"type": "string"},
                },
                "required": ["text", "target_language"],
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
    Task(prompt=f"What is the weather in {city}? Return in celsius.", expected_tool="get_weather", required_args=["location"])
    for city in [
        "Tokyo", "New York", "Paris", "London", "Berlin", "Sydney", "Toronto", "San Francisco", "Mumbai", "Singapore",
        "Seoul", "Mexico City", "Cairo", "Rome", "Madrid", "Buenos Aires", "Johannesburg", "Chicago", "Los Angeles", "Dubai",
        "Istanbul", "Bangkok", "Moscow", "Sao Paulo", "Lagos"
    ]
] + [
    Task(
        prompt="Send an email to boss@company.com saying the quarterly report is done.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email alice@example.com to confirm the 3pm meeting.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email bob@example.com that the deployment succeeded.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Send an email to hr@company.com requesting vacation from July 1-5.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email team@company.com with the summary: 'Sprint demo at 4pm'.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email support@example.com about a password reset issue.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Send an email to sales@example.com asking for the updated pricing sheet.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email ceo@company.com to reschedule today's 2pm meeting to 5pm.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email finance@example.com to approve the invoice #1234.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Send an email to ops@example.com that the server maintenance is complete.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email legal@example.com to review the new NDA template.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email hiring@example.com to proceed with the candidate offer.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email marketing@example.com to publish the blog post tomorrow.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email devrel@example.com asking for API rate limit increase.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email logistics@example.com to confirm the shipment tracking number.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email qa@example.com to rerun the regression suite on build 1.2.3.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email design@example.com for the updated logo assets.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email security@example.com to report a phishing attempt.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email billing@example.com to update the payment method.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email training@example.com to enroll in the new compliance course.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email it@example.com to request a new laptop.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email procurement@example.com to reorder office supplies.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email partners@example.com to confirm the partnership meeting next week.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email events@example.com to RSVP for the company offsite.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email hr@example.com to request an employment verification letter.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email payroll@example.com about a missing reimbursement.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email admin@example.com to book a conference room for tomorrow.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email onboarding@example.com to set up a new hire's accounts.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Email travel@example.com to change the flight to Monday.",
        expected_tool="send_email",
        required_args=["recipient", "body"],
    ),
    Task(
        prompt="Search the docs for guidance on rotating API keys.",
        expected_tool="search_docs",
        required_args=["query"],
    ),
    Task(
        prompt="Find documentation about our S3 backup policy.",
        expected_tool="search_docs",
        required_args=["query"],
    ),
    Task(
        prompt="Book a flight from SFO to JFK on March 14 for the CTO.",
        expected_tool="book_flight",
        required_args=["origin", "destination", "date"],
    ),
    Task(
        prompt="Book a flight from Toronto to London on May 2 for Sara.",
        expected_tool="book_flight",
        required_args=["origin", "destination", "date"],
    ),
    Task(
        prompt="Schedule a meeting with alice@example.com and bob@example.com tomorrow at 3pm.",
        expected_tool="schedule_meeting",
        required_args=["participants", "time"],
    ),
    Task(
        prompt="Set up a meeting with product@company.com next Monday at 10am.",
        expected_tool="schedule_meeting",
        required_args=["participants", "time"],
    ),
    Task(
        prompt="What's the latest price for AAPL?",
        expected_tool="get_stock_price",
        required_args=["symbol"],
    ),
    Task(
        prompt="Get me the current stock price for MSFT.",
        expected_tool="get_stock_price",
        required_args=["symbol"],
    ),
    Task(
        prompt="Translate 'Hello, how are you?' into Spanish.",
        expected_tool="translate_text",
        required_args=["text", "target_language"],
    ),
    Task(
        prompt="Translate 'Quarterly revenue beat expectations' to French.",
        expected_tool="translate_text",
        required_args=["text", "target_language"],
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

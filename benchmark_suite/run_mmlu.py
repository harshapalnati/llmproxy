#!/usr/bin/env python
"""
Lightweight MMLU harness for custom inference endpoints with leaderboard-like defaults.
- Modes:
  1) proxy_on: go through the reliability proxy (default behavior)
  2) proxy_off: go through the proxy but bypass RAPH logic via header `x-raph-mode: off`
  3) direct: call Positron directly
- Defaults to 5-shot prompting. Logprob scoring is optional (if your endpoint supports it).
"""

import argparse
import os
import re
import sys
from typing import Dict, Literal, Optional

import requests
from datasets import load_dataset


Mode = Literal["proxy_on", "proxy_off", "direct"]


def format_question(example) -> str:
    prompt = f"Question: {example['question']}\n"
    prompt += "Options:\n"
    prompt += f"A. {example['choices'][0]}\n"
    prompt += f"B. {example['choices'][1]}\n"
    prompt += f"C. {example['choices'][2]}\n"
    prompt += f"D. {example['choices'][3]}\n"
    prompt += "Answer with the single letter of the correct option (A, B, C, or D).\n"
    prompt += "Answer:"
    return prompt


def format_few_shot_prompt(dev_examples, target_example, shots: int) -> str:
    shots = min(shots, len(dev_examples))
    examples = list(dev_examples.select(range(shots)))
    parts = []
    for ex in examples:
        parts.append(format_question(ex))
        parts.append(f"{['A','B','C','D'][ex['answer']]}\n")
    parts.append(format_question(target_example))
    return "\n".join(parts)


def extract_answer(generated_text: str) -> Optional[str]:
    match = re.search(r"\b([A-D])\b", generated_text.upper())
    if match:
        return match.group(1)
    return None


def extract_answer_from_logprobs(logprobs: Dict) -> Optional[str]:
    """
    Attempt to score A/B/C/D from logprobs; pick the highest logprob token encountered.
    """
    if not logprobs:
        return None
    content = logprobs.get("content") or []
    best_token = None
    best_lp = float("-inf")
    target_tokens = {"A", "B", "C", "D"}

    for item in content:
        token = item.get("token")
        lp = item.get("logprob")
        if token in target_tokens and lp is not None and lp > best_lp:
            best_lp = lp
            best_token = token
        for tlp in item.get("top_logprobs") or []:
            token2 = tlp.get("token")
            lp2 = tlp.get("logprob")
            if token2 in target_tokens and lp2 is not None and lp2 > best_lp:
                best_lp = lp2
                best_token = token2
    return best_token


def query_model(prompt: str, mode: Mode, model: str, temperature: float, use_logprobs: bool) -> Dict:
    proxy_url = os.getenv("PROXY_URL", "http://localhost:9000/v1")
    positron_url = os.getenv("POSITRON_URL", "https://api.positron.ai/v1")
    api_key = os.getenv("POSITRON_KEY")
    if not api_key:
        raise RuntimeError("Missing POSITRON_KEY in environment.")

    if mode in ("proxy_on", "proxy_off"):
        url = f"{proxy_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        if mode == "proxy_off":
            headers["x-raph-mode"] = "off"
    else:
        url = f"{positron_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer with a single letter (A, B, C, or D).",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    if use_logprobs:
        payload["logprobs"] = {"top_logprobs": 4}

    resp = requests.post(url, json=payload, headers=headers, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Request failed ({resp.status_code}): {resp.text}")

    data = resp.json()
    if not isinstance(data, dict) or "choices" not in data:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return data


def run_mmlu(
    subject: str,
    num_samples: int,
    mode: Mode,
    model: str,
    temperature: float,
    shots: int,
    use_logprobs: bool,
) -> float:
    print(
        f"--- Running MMLU: {subject} | mode={mode} | model={model} | shots={shots} | logprobs={use_logprobs} ---"
    )
    test_data = load_dataset("cais/mmlu", subject, split="test")
    dev_data = load_dataset("cais/mmlu", subject, split="dev")
    score = 0
    total = 0

    for i, item in enumerate(test_data):
        if i >= num_samples:
            break

        prompt = (
            format_few_shot_prompt(dev_data, item, shots)
            if shots > 0 and len(dev_data) > 0
            else format_question(item)
        )
        correct_letter = ["A", "B", "C", "D"][item["answer"]]

        data = query_model(prompt, mode=mode, model=model, temperature=temperature, use_logprobs=use_logprobs)
        choice = data["choices"][0]

        predicted_letter = None
        if use_logprobs and choice.get("logprobs"):
            predicted_letter = extract_answer_from_logprobs(choice["logprobs"])
        if not predicted_letter:
            predicted_letter = extract_answer(choice["message"]["content"])

        raw_text = choice["message"]["content"]
        if predicted_letter == correct_letter:
            score += 1
            print(f"✅ Q{i+1}: Correct")
        else:
            print(f"❌ Q{i+1}: Wanted {correct_letter}, got {predicted_letter} (Raw: {raw_text})")

        total += 1

    accuracy = (score / total) * 100 if total else 0
    print(f"\nFinal Score for {subject} ({mode}): {accuracy:.2f}%")
    return accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight MMLU benchmark against a custom endpoint.")
    parser.add_argument("--subject", default="college_computer_science", help="MMLU subject split to evaluate.")
    parser.add_argument("--num-samples", type=int, default=5, help="How many test samples to run.")
    parser.add_argument(
        "--mode",
        choices=["proxy_on", "proxy_off", "direct"],
        default="proxy_on",
        help="proxy_on (reliability on), proxy_off (bypass proxy logic), direct (call Positron directly).",
    )
    parser.add_argument("--model", default=os.getenv("MMLU_MODEL", "llama-3.1-8b-instruct-good-tp2"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--shots", type=int, default=5, help="Number of few-shot examples (0 for zero-shot).")
    parser.add_argument(
        "--use-logprobs",
        action="store_true",
        help="Enable logprob scoring (only if your endpoint supports it).",
    )

    args = parser.parse_args()

    try:
        run_mmlu(
            subject=args.subject,
            num_samples=args.num_samples,
            mode=args.mode,  # type: ignore[arg-type]
            model=args.model,
            temperature=args.temperature,
            shots=args.shots,
            use_logprobs=args.use_logprobs,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

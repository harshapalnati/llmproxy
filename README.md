Rust reliability proxy for tool calls (RAPH loop) sitting in front of Positron.

## Prereqs
- Rust toolchain (stable) with `cargo`
- Optional: Python 3.10+ for the benchmark scripts

## Run the proxy
```bash
# in repo root
cargo run --bin proxy_server
```

Environment knobs (can go in `.env`):
- `POSITRON_URL` (default `http://localhost:8080/v1`)
- `POSITRON_KEY` (default `sk-placeholder`)
- `MAX_RETRIES` (default `3`)
- `UPSTREAM_TIMEOUT_SECS` (default `20`) timeout per Positron attempt (connect + full request)
- `PROXY_PORT` (default `9000`)

Send OpenAI-compatible traffic to `http://localhost:9000/v1`. To bypass reliability logic: header `x-raph-mode: off` (or `false`) or URL query `?use_raph=false`. Default is reliability ON.

Quick cURL check (replace `sk-...`):
```bash
curl -X POST http://localhost:9000/v1/chat/completions \
  -H "content-type: application/json" \
  -H "authorization: Bearer sk-placeholder" \
  -d '{
    "model": "dummy",
    "messages": [
      {"role": "user", "content": "hi"}
    ]
  }'
```

## Binary
- `proxy_server`: injects XML tool instructions, repairs/validates tool calls, retries, and rewrites responses into OpenAI-style `tool_calls`. Supports bypass via header/URL.

## Dev loop
- Lint/format: `cargo fmt`
- Smoke tests: `cargo test` (no suite yet; runs quickly)

## Benchmark suite (MMLU)
- Script: `benchmark_suite/run_mmlu.py`
- Env: `POSITRON_KEY` (required), `PROXY_URL` (default `http://localhost:9000/v1`), `POSITRON_URL` (default `https://api.positron.ai/v1`), optional `MMLU_MODEL` (default `llama-3.1-8b-instruct-good-tp2`).
- Modes:
  - `proxy_on` (default): reliability loop on
  - `proxy_off`: bypass proxy logic via `x-raph-mode: off`
  - `direct`: call Positron directly

Example:
```bash
python benchmark_suite/run_mmlu.py --subject global_facts --num-samples 5 --mode proxy_on
python benchmark_suite/run_mmlu.py --subject global_facts --num-samples 5 --mode proxy_off
python benchmark_suite/run_mmlu.py --subject global_facts --num-samples 5 --mode direct
```

### Tool-call mini-benchmark
- Script: `benchmark_suite/run_toolbench.py`
- Modes: `proxy_on` (default), `proxy_off`, `direct`
- Tasks: built-in small set of tool-use prompts; or provide JSONL via `--tasks` with `prompt`, `expected_tool`, `required_args`.

Example:
```bash
python benchmark_suite/run_toolbench.py --mode proxy_on --limit 5
python benchmark_suite/run_toolbench.py --mode direct --limit 5
```

Rust reliability proxy for tool calls (RAPH loop) sitting in front of Positron.

## Binary
- `proxy_server`: injects XML tool instructions, repairs/validates tool calls, retries, and rewrites responses into OpenAI-style `tool_calls`. Supports bypass via header/URL.

## Config
- `POSITRON_URL` (default `http://localhost:8080/v1`)
- `POSITRON_KEY` (default `sk-placeholder`)
- `MAX_RETRIES` (default `3`)
- `PROXY_PORT` (default `9000`)

## Bypass switch
- Header: `x-raph-mode: off` (or `false`) disables reliability and forwards directly (streams honored).
- URL: `?use_raph=false` does the same for raw HTTP requests.

## Run
```bash
cargo run --bin proxy_server
```

Point your OpenAI-compatible client at `http://localhost:9000/v1`. To bypass logic: add `x-raph-mode: off` header or `?use_raph=false` query param. Default is reliability ON.

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

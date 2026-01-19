# Toolbench proxy benchmarking (llama-3.1-8b-instruct-good-tp2)

- Proxy config: `POSITORON_URL=https://api.positron.ai/v1`, `MAX_RETRIES=8`, port 9000; fresh log in `/tmp/raph_proxy.log`.
- Benchmark script: `benchmark_suite/run_toolbench.py` from venv; 30 tasks (weather + email prompts).
- Date: latest run in this session.

## Results
- `proxy_on` (reliability enabled): 21 / 30 tasks succeeded (70%). Runtime ~10.4 minutes.
- `proxy_off` (bypass, raw model): 0 / 30 tasks succeeded. Runtime ~36 seconds.

## Retry behavior (proxy_on run)
- Proxy recorded 159 retry attempts while repairing/validating tool calls.
- Typical failure: model emitted unquoted keys (`{location: "Tokyo"}`), triggering repair errors like `parse error: key must be a string at line 1 column 2`.
- Retries stretched per-task latency; some tasks still failed after 8 attempts (`Max retries exceeded`).

## Notes
- The reliability loop materially improves correctness vs bypassing, but latency is high when the model keeps emitting malformed JSON.
- Log file with all attempts: `/tmp/raph_proxy.log`.
- Commands used:
  - `python benchmark_suite/run_toolbench.py --mode proxy_on  --model llama-3.1-8b-instruct-good-tp2 --limit 30`
  - `python benchmark_suite/run_toolbench.py --mode proxy_off --model llama-3.1-8b-instruct-good-tp2 --limit 30`

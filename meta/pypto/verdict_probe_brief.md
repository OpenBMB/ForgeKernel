## Ascend verdict experiments

Up to {MAX_VERDICT_PROBE_ROUNDS} driver-run experiments can test this diagnosis.
Write a self-contained `notes/<name>.py` and end with only
`PROBE_REQUEST: notes/<name>.py`. Pre-register each hypothesis and threshold in
the script. Print `PROBE PASS max_diff=<finite reading>` or
`PROBE FAIL max_diff=<finite reading>` per hypothesis; non-numeric compile or
launch contracts use `max_diff=n/a`. All contracts must pass for certification.
A diagnostic FAIL remains measured evidence. No verdict means it did not run.

For a topology price use the seeded `notes/ascend_probe.py` with an exact-edit
JSON specification: base and variant are independently built and judged by the
frozen full-case CANN harness, alternating across repeats. The tool records
casewise times, geometric means, spreads and the candidate SHA. It never changes
the working kernel. Do not use CUDA/SASS/timeline helpers. A synthetic probe's
speed is not an official operator gate reading. Cite measured numeric gain via
`reopen_price: <±x%> via probe:notes/<name>.py#<contract index>`.

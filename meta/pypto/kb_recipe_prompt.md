# Certify one reusable PyPTO / Ascend hardware mechanism

Read `{META_MILESTONE_PATH}`, `{KB_DIR}/TEMPLATE.md` and
`{KB_READ_DIR}/index.md`. Address the appended G3 diagnosis. Use permitted local
PyPTO/PTO API sources; oracle `{ORACLE}` is only readable if the run permits it.
Never put an operator algorithm, source paraphrase, reference symbol, shape-
specific implementation or measured oracle footprint in the KB.

Write `{KB_READ_DIR}/<mechanism>.md` with Purpose, When needed, Minimal runnable
example, Gotchas, Sources and certification scope. Use only generic synthetic
inputs and PyPTO NPU operations. Record no-go conditions and what must change
before retry. Put experiment narrative in `{PROVENANCE_DIR}/<mechanism>.md`;
agent-readable provenance cites only public docs/API symbols and probe verdicts.

Write a standalone `{WORKSPACE}/notes/<mechanism>.py`. Exercise the mechanism
on the real NPU, synchronize, compute an independent finite error, then print
`PROBE PASS max_diff=<error>` or `PROBE FAIL max_diff=<error>`. For a compile/
launch-only claim print `max_diff=n/a`; that certifies no numerical or speed
claim. All contract lines must pass. Never self-certify without a driver job.

End with `PROBE_REQUEST: notes/<mechanism>.py`. On failure, fix the probe or
claim and request again. No verdict is a build/launch failure, not numerical
failure. Only after the driver returns PASS update the index to Verified with
job, probe and scope, then end with:
`RECIPE_CERTIFIED: hw_kb/pypto/{KB_ARCH}/<mechanism>.md`

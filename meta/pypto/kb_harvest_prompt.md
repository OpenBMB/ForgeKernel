# PyPTO finish-time knowledge harvest

Read `{META_MILESTONE_PATH}`, `{KB_DIR}/TEMPLATE.md`, `{KB_READ_DIR}/index.md`
and the appended measured probe history from `{WORKSPACE}`. This turn runs NO
NPU jobs. Only mechanisms already certified by a completed NPU probe may enter
the KB. A correct/fast operator, an msprof signature or an agent's explanation
alone is not certification of every mechanism it contains.

For each genuinely reusable hardware/API lesson, write or update
`{KB_READ_DIR}/<mechanism>.md`: Purpose / When needed / Minimal runnable example
(extracted from the tested probe) / Gotchas / Sources. Include architecture,
software scope, trigger symptom, tested negative controls, no-go conditions and
retry preconditions. Index evidence is job + probe path + measured verdict.
Do not put operator names/algorithms, oracle symbols/paraphrases, run IDs or
commit hashes into any KB file. Narrative belongs only in
`{PROVENANCE_DIR}/<mechanism>.md`. Reference `{ORACLE}` remains subject to the
run's disclosure policy. Unknown mechanisms are skipped, not falsely certified.

End with one `RECIPE_RECORDED: hw_kb/pypto/{KB_ARCH}/<mechanism>.md` per updated
recipe, or `NO_RECIPE: <reason>`. Never send PROBE_REQUEST in this turn.

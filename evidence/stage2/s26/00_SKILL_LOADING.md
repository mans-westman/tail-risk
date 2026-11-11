# 00 — Skill loading evidence (runtime, not filesystem)

Skill name: `tail-risk-audit`

## Verbatim runtime excerpt

The runtime tells a running agent when a skill has become available. Copied without editing
from the runtime's own status attachment in the repair session (home directory written as `~`):

```
Skill environment status update. Invoke relevant skills only when needed for the current task.
Newly available skills:
0. `tail-risk-audit`: Run and audit the tailrisk VaR/ES comparison workflow on one daily return
series. Use when the user wants to compare next-session value-at-risk and expected shortfall
models, run or verify a tailrisk run directory, ingest a declared returns CSV, score saved
forecasts, render the evidence report, or explain what a tailrisk run does and does not
establish.
```

That paragraph is the registry's own text, not the agent's summary of it. A directory on disk
does not produce this notice; a loaded entry does. The trial ran before this excerpt was
captured, against the same installed entry — the install predates the trial and was not touched
between them.

## Re-runnable registry evidence

The runtime's own inventory scanner, run again over the live skills root:

```
$ python .tmp/check_live_skill.py
inventory dir: .tmp/skills_view
count: 15
fingerprint: 513d926798759e46c9dcae1dc1de35827c60e9cab7c71df77ce4dd45b4f284c4
tail-risk-audit: discovered source='local'
  skill_file: ~/.jiuwenswarm/agent/workspace/skills/tail-risk-audit/SKILL.md
  on disk:    ~/.jiuwenswarm/agent/workspace/skills/tail-risk-audit/SKILL.md
  same file:  True
  description chars: 356
  content_hash: fb440e93d14054b96760a3dfccfe3bd703c243b38fa102462fdb169375d38155
  frontmatter keys: ['description', 'name']
```

| Item | Evidence |
|---|---|
| Installed source directory | `~/.jiuwenswarm/agent/workspace/skills/tail-risk-audit/` — `SKILL.md`, `check_references.py`, `references/{DATA_CONTRACT,ENVIRONMENT,METHODS,MODEL_ADAPTER,REPORTING}.md`, byte-identical to `skills/tail-risk-audit/` in the repository (`diff -r` clean). |
| Reference integrity | `.venv/bin/python skills/tail-risk-audit/check_references.py` → `reference check passed: 5 pages identical to docs/` (exit 0). In an installed copy it reports "there is no `docs/` to compare" and exits 0, which is not drift. |
| Skill body returned to the trial agent | The trial session reported that `skill_tool(skill_name="tail-risk-audit")` returned the whole `SKILL.md` body — preconditions, seven workflow steps, guardrails, references table — with no error. That call is not archived; the notice above and the installed tree are. |

## Approval gate

| Item | Status |
|---|---|
| Explicit authorisation before the trial ran | Given by the project owner in the commissioning session. **Its wording is not reproduced here**, because no transcript of that session was archived in the repository and a reconstruction would be invention presented as evidence. |
| What is recorded instead | The trial's observable boundary: it created only the files listed in `01_ENVIRONMENT.txt`, left `examples/bring_your_model/ewma_gaussian.py` unmodified (no diff), and staged, committed and pushed nothing. That matches what was authorised: a trial, not a release. |
| This repair round | Authorised in writing; the instruction is quoted verbatim at the top of `04_REPAIR.md`. |

The distinction matters: loading is evidenced by the runtime notice and the installed tree;
the approval is evidenced only by the project owner's instruction, and where that instruction is
not archived this record says so rather than implying a transcript exists.

## What the skill supplied, and what it did not

Supplied: confirm the interpreter, read the config before running, use a fresh output
directory, run → verify → evaluate → report, and never translate `unavailable` coverage into a
pass. Not supplied: any instruction on integrating an externally supplied model. That came from
`docs/MODEL_ADAPTER.md` and `examples/bring_your_model/README.md`, which the skill's references
table points at only for "the user wants their own model".

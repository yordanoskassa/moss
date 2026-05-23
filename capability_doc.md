# MOSS Self-Evolution — Capability Document

## Overview

MOSS (Model-Originated Source Surgery) enables this agent to evolve its own harness
code — routing, hooks, state management, and dispatch — through source-level rewriting.
When failures are detected in task execution, MOSS can analyze, plan, implement, and
deploy fixes automatically.

## Available Commands

All commands are accessed via `moss evo <subcommand>`:

### Observation Commands

| Command | Description |
|---------|-------------|
| `moss evo status` | Show current evolution state (running, idle, completed) |
| `moss evo batches` | List all sealed failure batches |
| `moss evo batch <id>` | Inspect a specific batch's chunks and metadata |

### Control Commands

| Command | Description |
|---------|-------------|
| `moss evo start <batch-id> [--depth light\|standard\|deep]` | Start an evolution run against a batch |
| `moss evo stop` | Halt the current evolution run |
| `moss evo restart <evo-id>` | Restart a failed or stopped evolution |
| `moss evo apply <image-tag>` | Deploy a candidate image via the swap supervisor |

### Scan Commands

| Command | Description |
|---------|-------------|
| `moss evo flag <session-id>` | Scan a single session for failure chunks |
| `moss evo catch-up` | Scan all sessions from their last cursors |

## Depth Dial

The `--depth` option controls how thorough the evolution process is:

- **light**: 2 iterations max, 1 trial per task, quick convergence
- **standard** (default): 5 iterations, 3 trials per task, balanced
- **deep**: 10 iterations, 5 trials per task, thorough exploration

## Evolution Pipeline

Each iteration runs 7 stages:

1. **Locate** — Diagnose failures from traces and batch chunks
2. **Plan** — Produce a fix specification (files, functions, logic changes)
3. **Plan Review** — Quality gate: approve, reject-off-target, or reject-too-narrow
4. **Implement** — Write the fix as a git commit
5. **Code Review** — Diff quality gate
6. **Task Evaluate** — Score keypoints (strong/adequate/weak/missing)
7. **Verdict** — Convergence decision (converged / need-more-work / fundamental-limit)

## Operational Rules

1. **Do not start evolution during active user sessions** — wait for idle periods.
2. **Always check `moss evo status` before starting** — only one evolution runs at a time.
3. **Review batch contents before evolving** — use `moss evo batch <id>` to verify the
   failure chunks are representative.
4. **Start with `--depth light`** for exploratory runs, escalate to `standard` or `deep`
   for production fixes.
5. **After convergence**, use `moss evo apply <image-tag>` to deploy. The swap supervisor
   handles health probes and automatic rollback.
6. **If evolution reaches FUNDAMENTAL_LIMIT**, the failures cannot be fixed by source
   rewriting alone — escalate to human operators.

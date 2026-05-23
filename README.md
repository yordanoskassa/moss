# MOSS: Self-Evolution through Source-Level Rewriting

MOSS is a system from the paper [arXiv:2605.22794](https://arxiv.org/abs/2605.22794) that enables deployed autonomous agents to rewrite their own source code when they detect failures. Unlike existing self-evolving agents that only modify prompts, skills, or memory, MOSS targets the **agent harness itself** — routing, hooks, state management, dispatch — which is unreachable from text-mutable artifacts.

The paper's repo (`github.com/dav-joy-thon/MOSS`) is not public, so this implementation is built from scratch using the paper's detailed architecture as specification. It validates against the same ClawEval benchmark (baseline ~0.25 → post-evolution ~0.61).

## Stack

- **Runtime**: Python 3.12+ / FastAPI / asyncio / Docker
- **Substrate**: OpenClaw (TypeScript/Node.js)
- **Storage**: File-based JSONL
- **Coding Agent**: Claude Code (subprocess runner)

## Architecture

MOSS consists of four major subsystems:

1. **Auto-Scan Engine** — Monitors session transcripts, slices them into chunks, and groups chunks into sealed batches when a failure threshold is reached.

2. **7-Stage Evolution Pipeline** — Each iteration runs:
   - **Locate** — Diagnose failures from traces and batch chunks
   - **Plan** — Produce a fix specification (files, functions, logic changes)
   - **Plan Review** — Quality gate (multi-round: approve / reject-off-target / reject-too-narrow)
   - **Implement** — Write the fix as a git commit via the coding agent
   - **Code Review** — Diff quality gate (multi-round)
   - **Task Evaluate** — Score 4–7 keypoints per task (strong / adequate / weak / missing)
   - **Verdict** — Convergence decision with plateau guard

3. **Docker Trial Workers + Swap Supervisor** — Builds candidate images, runs ephemeral trial containers, and manages zero-downtime container replacement with health probes and automatic rollback.

4. **Host Daemon + Gateway** — Unix-socket RPC daemon for background services, FastAPI gateway for HTTP control, and webhook receivers for lifecycle events.

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

All commands are accessed via `moss evo <subcommand>`:

```bash
# Observation
moss evo status              # Show current evolution state
moss evo batches             # List all sealed failure batches
moss evo batch <id>          # Inspect a specific batch

# Control
moss evo start <batch-id>    # Start an evolution run
moss evo stop                # Halt the current evolution
moss evo restart <evo-id>    # Restart a failed evolution
moss evo apply <image-tag>   # Deploy a candidate image

# Scanning
moss evo flag <session-id>   # Scan a single session
moss evo catch-up            # Scan all sessions from cursors
```

### Depth Dial

Control evolution thoroughness with `--depth`:

| Depth | Max Iterations | Trials/Task | Plateau Threshold |
|-------|---------------|-------------|-------------------|
| light | 2 | 1 | 1 |
| standard | 5 | 3 | 3 |
| deep | 10 | 5 | 5 |

## Running the Daemon

```bash
moss-daemon
```

Starts the Unix-socket RPC server, swap supervisor, and periodic auto-scan.

## Docker

```bash
docker compose up
```

Starts the gateway container and daemon with shared volumes.

## Tests

```bash
pytest tests/ -v
```

## Configuration

See [`moss.toml`](moss.toml) for all configuration options including depth dials, runner settings, swap probe parameters, and gateway/daemon network config.

## License

[MIT](LICENSE)

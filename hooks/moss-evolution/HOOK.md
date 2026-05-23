# MOSS Evolution Hook

## Metadata

- **name**: moss-evolution
- **version**: 0.1.0
- **events**:
  - `agent:bootstrap` — Inject MOSS capability documentation into system prompt
  - `message:received` — Route evolution webhook payloads as system messages

## Description

This hook integrates the MOSS self-evolution system with the OpenClaw agent substrate.
On agent bootstrap, it injects a one-paragraph pointer to the capability document
so the agent is aware of its self-evolution abilities. When evolution lifecycle events
occur (convergence, failure, swap completion), the webhook payloads are translated
into system messages for the agent.

## Configuration

The hook reads `capability_doc.md` from the MOSS bind-mount path (default: `/opt/moss/capability_doc.md`).
The gateway URL defaults to `http://localhost:8420`.

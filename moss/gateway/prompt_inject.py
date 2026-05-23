"""System prompt injection for capability documentation."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_CAP_DOC_PATH = Path("/opt/moss/capability_doc.md")
_FALLBACK_CAP_DOC_PATH = Path(__file__).parent.parent.parent / "capability_doc.md"

INJECTION_TEMPLATE = """\

---
## MOSS Self-Evolution Capability

This agent has access to MOSS, a self-evolution system that can rewrite the agent's \
own source code when failures are detected. For details on available commands and \
operational rules, refer to the capability document.

To interact with MOSS, use the `moss evo` CLI with these subcommands:
  status, batches, batch, start, stop, restart, apply, flag, catch-up

Run `moss evo --help` for usage details.
---
"""


def load_capability_doc(path: Path | None = None) -> str | None:
    """Load the capability document from disk."""
    candidates = [path] if path else [_DEFAULT_CAP_DOC_PATH, _FALLBACK_CAP_DOC_PATH]

    for p in candidates:
        if p and p.exists():
            try:
                return p.read_text()
            except OSError as e:
                logger.warning("Could not read capability doc at %s: %s", p, e)

    logger.warning("No capability document found")
    return None


def build_injection(cap_doc: str | None = None) -> str:
    """Build the system prompt injection text."""
    if cap_doc:
        return f"\n---\n## MOSS Capability Document\n\n{cap_doc}\n---\n"
    return INJECTION_TEMPLATE


def inject_into_prompt(system_prompt: str, cap_doc_path: Path | None = None) -> str:
    """Inject MOSS capability information into a system prompt."""
    cap_doc = load_capability_doc(cap_doc_path)
    injection = build_injection(cap_doc)
    return system_prompt + injection

/**
 * MOSS Evolution Hook Handler
 *
 * Integrates MOSS self-evolution with the OpenClaw agent substrate.
 * - On agent:bootstrap: injects capability doc pointer into system prompt
 * - On message:received: translates evolution webhooks into system messages
 */

import { readFileSync, existsSync } from "fs";
import { join } from "path";

// Capability doc paths (in order of preference)
const CAP_DOC_PATHS = [
  "/opt/moss/capability_doc.md",
  join(process.cwd(), "capability_doc.md"),
];

const GATEWAY_URL = process.env.MOSS_GATEWAY_URL || "http://localhost:8420";

/**
 * Load the capability document from disk.
 */
function loadCapabilityDoc(): string | null {
  for (const p of CAP_DOC_PATHS) {
    if (existsSync(p)) {
      try {
        return readFileSync(p, "utf-8");
      } catch {
        // Continue to next path
      }
    }
  }
  return null;
}

/**
 * Build the system prompt injection.
 */
function buildInjection(): string {
  const doc = loadCapabilityDoc();
  if (doc) {
    return `\n---\n## MOSS Capability Document\n\n${doc}\n---\n`;
  }
  return `
---
## MOSS Self-Evolution Capability

This agent has access to MOSS, a self-evolution system that can rewrite the agent's
own source code when failures are detected. For details on available commands and
operational rules, refer to the capability document.

To interact with MOSS, use the \`moss evo\` CLI with these subcommands:
  status, batches, batch, start, stop, restart, apply, flag, catch-up

Run \`moss evo --help\` for usage details.
---
`;
}

/**
 * Handle agent:bootstrap event.
 * Injects MOSS capability info into the system prompt.
 */
export function onAgentBootstrap(context: {
  systemPrompt: string;
}): { systemPrompt: string } {
  const injection = buildInjection();
  return {
    systemPrompt: context.systemPrompt + injection,
  };
}

/**
 * Handle message:received event.
 * Checks for pending MOSS webhook messages and injects them.
 */
export async function onMessageReceived(context: {
  message: unknown;
}): Promise<{ systemMessages?: string[] }> {
  try {
    const response = await fetch(`${GATEWAY_URL}/evo/webhooks/pending`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(5000),
    });

    if (!response.ok) {
      return {};
    }

    const data = (await response.json()) as { messages?: Array<{ content: string }> };
    const messages = data.messages || [];

    if (messages.length === 0) {
      return {};
    }

    return {
      systemMessages: messages.map(
        (m) => `[MOSS Evolution] ${m.content}`
      ),
    };
  } catch {
    // Gateway unavailable — silently skip
    return {};
  }
}

// Event router
export default function handler(
  event: string,
  context: Record<string, unknown>
): unknown {
  switch (event) {
    case "agent:bootstrap":
      return onAgentBootstrap(context as { systemPrompt: string });
    case "message:received":
      return onMessageReceived(context as { message: unknown });
    default:
      return {};
  }
}

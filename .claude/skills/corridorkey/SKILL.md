---
name: corridorkey
description: "Use local CorridorKey for screen keying, alpha hints, or masked green-background compositing. Load the canonical repository workflow."
---

# CorridorKey adapter

Read the canonical skill at
[.agents/skills/corridorkey/SKILL.md](../../../.agents/skills/corridorkey/SKILL.md)
from the repository root before processing video. The link is relative to this
file's directory. Follow its relevant runtime and Resolve references.

Use this checkout's `bootstrap.py` and `run.py`. The runtime configuration is
machine-local and ignored by Git. Do not substitute the upstream cu128 installer
for a P600 profile. Use the canonical helper scripts; do not copy or maintain a
second engine/workflow in this adapter directory.

This adapter registers `/corridorkey` in Claude Code. It supplies no local tool
connection to a cloud-only chat session and does not create a GUI or MCP server.

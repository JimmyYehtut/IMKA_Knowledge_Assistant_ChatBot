---
description: Run the live diagram eval against the running stack and explain any failures
argument-hint: "[case-id substring]"
---

Use the `rag-eval` agent to run the live diagram eval. If an argument was given (`$ARGUMENTS`), pass it as `--only $ARGUMENTS` so only matching cases run.

The eval needs the stack running (`start.ps1`) and spends OpenAI tokens. If the API health check fails, say so and stop.

Relay the agent's pass count and, per failure, the stage at fault and the suggested fix. Do not apply fixes unless I ask.

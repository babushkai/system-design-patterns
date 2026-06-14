# LLM Agent Fundamentals

## TL;DR

An agent is a model using tools in a loop against an environment. The model supplies reasoning; the **harness** — the loop, tool definitions, context management, permissions, and sandbox you build around it — determines how much of that reasoning turns into useful work. Modern agents use native tool calling (typed JSON schemas, parallel calls) rather than text-parsed prompts, treat the context window as working memory backed by files and compaction, verify their own work against ground truth (tests, linters, screenshots), and run inside permission-gated sandboxes. Design the environment and the feedback loops first; the model is the most replaceable component.

---

## What Makes an Agent Different from a Chatbot?

```mermaid
graph LR
    subgraph CHATBOT["Chatbot / Workflow"]
        UI1["User Input"] --> LLM1["LLM"] --> R1["Response"]
    end

    subgraph AGENT["Agent"]
        UG["Goal"] --> M["Model decides<br/>next action"]
        M -->|tool call| ENV["Environment<br/>(files, shell, APIs, browser)"]
        ENV -->|tool result| M
        M -->|no more tool calls| DONE["Final answer"]

        HARNESS["Harness:<br/>loop, context mgmt,<br/>permissions, sandbox"] -.->|mediates| M
        HARNESS -.->|mediates| ENV
    end
```

A chatbot maps one input to one output. A **workflow** chains LLM calls along a code path you wrote. An **agent** lets the model direct its own process: it decides which tool to call next based on what the last tool returned, and it keeps going until the goal is met or the harness stops it. That autonomy is the value and the risk — agents handle tasks you couldn't enumerate steps for, and they fail in ways you didn't enumerate either.

| Aspect | Chatbot | Workflow | Agent |
|--------|---------|----------|-------|
| Control flow | None | Your code | The model |
| Actions | Text only | Predefined LLM calls | Tools chosen at runtime |
| Steps | 1 | Fixed | Open-ended, bounded by harness |
| Failure mode | Bad answer | Bad step output | Compounding drift across steps |
| Cost profile | Predictable | Predictable | Variable — budget in the harness |
| Right for | Q&A | Decomposable, known tasks | Open-ended tasks with verifiable outcomes |

Start with the simplest form that solves the problem. Most production "agent" systems are workflows, and should be — see [Orchestration Patterns](./02-orchestration-patterns.md) for the taxonomy.

## The Three Components

```mermaid
graph TD
    MODEL["MODEL<br/>Reasoning, tool selection,<br/>extended thinking"]
    HARNESS["HARNESS<br/>Loop, system prompt, tool schemas,<br/>context lifecycle, permissions,<br/>checkpointing, telemetry"]
    ENVIRONMENT["ENVIRONMENT<br/>Filesystem, shell, APIs,<br/>browser, sandbox boundary"]

    HARNESS --> MODEL
    HARNESS --> ENVIRONMENT
    MODEL <-.->|tool calls / results| ENVIRONMENT
```

- **Model.** Frontier models are post-trained specifically for agentic tool use: they natively interleave reasoning with tool calls, recover from errors, and sustain multi-hour tasks. Capability differences between models matter, but a mediocre harness wastes a frontier model — public coding benchmarks always report a *model + harness* pair, never a model alone.
- **Harness.** Everything between the model and the world. This is where most engineering effort goes; see [Harness Engineering](./09-harness-engineering.md) for the full treatment.
- **Environment.** What the agent can observe and change. The richer and more inspectable the environment (a real shell, a real filesystem, real test suites), the better the agent's feedback loops. Design environments so that progress is *observable* — an agent that can run the tests doesn't need to guess whether its change worked.

---

## The Agent Loop

The 2023-era pattern — prompt the model to emit `Thought: / Action: / Action Input:` text and parse it with regexes — is obsolete. Every major API exposes **native tool calling**: you declare tools as JSON Schema, the model returns typed tool-call blocks, and you return results as structured messages. No parsing, no format drift, parallel calls for free.

```python
import anthropic

client = anthropic.Anthropic()

TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the project sandbox. "
                       "Returns stdout and stderr, truncated to 50KB.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Command to execute"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file. Fails if the string "
                       "is not found or matches more than once.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
            },
            "required": ["path", "old", "new"],
        },
    },
]

def agent_loop(task: str, max_turns: int = 50) -> str:
    messages = [{"role": "user", "content": task}]

    for _ in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8192,
            system=SYSTEM_PROMPT,          # stable across turns: prompt-cache friendly
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return next(b.text for b in response.content if b.type == "text")

        # Append the assistant turn verbatim, then execute every tool call
        # in it (the model may request several in parallel).
        messages.append({"role": "assistant", "content": response.content})
        results = [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": execute_tool(block.name, block.input),
            }
            for block in response.content
            if block.type == "tool_use"
        ]
        messages.append({"role": "user", "content": results})

    raise RuntimeError("max_turns exceeded — task did not converge")
```

The OpenAI equivalent uses `tools=[{"type": "function", ...}]` and `tool_calls` on the response; the shape of the loop is identical. The legacy `functions` / `function_call` parameters are deprecated.

What the production version adds on top of this skeleton:

1. **Permission gate** before `execute_tool` — classify actions as read / write / irreversible and require approval for the last class.
2. **Token budget accounting** — track context usage every turn; trigger compaction before overflow (see [Context Management](./08-context-management.md)).
3. **Checkpointing** — persist `messages` so a crashed or interrupted run resumes instead of restarting.
4. **Telemetry** — record every turn as a trace span: tool name, latency, tokens, cache hit rate.
5. **Streaming and interruption** — surface partial output and let a human steer mid-run.

### Extended thinking

Reasoning-capable models can emit internal thinking tokens before acting, and *interleave* thinking between tool calls — reflect on a result before choosing the next action. This replaced most prompt-level reasoning scaffolds (Chain-of-Thought, Tree-of-Thought); you control it with a thinking budget parameter rather than prompt tricks. Spend the budget where verification is hard and steps are irreversible; keep it low for mechanical tool-use sequences.

---

## Tools

Tools are the agent's API to the world. Tool design is prompt engineering with a type system — the model reads your schemas and descriptions the way a new hire reads your docs, and ambiguity costs you real tokens and wrong calls.

### Design principles

1. **Few, orthogonal tools beat many overlapping ones.** Every tool competes for the model's attention in every turn. If two tools can accomplish the same thing, the model will sometimes pick the worse one. Consolidate (`search_logs(filters)` rather than five log tools).
2. **Descriptions are micro-prompts.** State what the tool does, when to use it over its neighbors, what it returns, and its failure modes. The `edit_file` description above tells the model *how to avoid* the two common failure cases.
3. **Token-efficient outputs.** Return what the model needs to decide the next step — truncate, paginate, summarize. A tool that dumps a 200KB JSON blob into context does more damage than one that errors.
4. **Errors that teach.** `"File not found: src/uesr.py — did you mean src/user.py?"` lets the model self-correct in one turn. A bare stack trace often costs three.
5. **Idempotent and safe to retry** wherever possible — the loop *will* retry.

### General-purpose vs. structured tools

The highest-leverage tools in practice are the general-purpose ones: **a shell, file read/write/edit, and code execution**. A bash tool subsumes hundreds of specialized tools, and writing a script is often more token-efficient than chaining ten tool calls — the "code execution as tool use" pattern has the agent *write a program* that calls APIs in a loop instead of making each call through the context window. Add structured tools where types and guardrails matter: payments, ticket updates, anything irreversible.

### MCP: the integration standard

The Model Context Protocol (MCP) standardizes how tools, resources, and prompts are exposed to agents — an MCP server wraps a system (GitHub, Postgres, Slack, a browser) once, and any MCP-capable harness can use it. Treat MCP servers as third-party dependencies: review what they expose, pin versions, and remember that every connected server's tool descriptions enter your prompt (use deferred loading / tool search when the catalog is large). Coverage in depth: [Coding Agent Tool Design](../18-compound-engineering/02-coding-agent-tool-design.md).

```mermaid
graph LR
    AGENT["Agent harness"] -->|MCP client| S1["MCP server: GitHub"]
    AGENT -->|MCP client| S2["MCP server: Postgres"]
    AGENT -->|MCP client| S3["MCP server: internal APIs"]
    AGENT -->|built-in| T1["bash / files / code exec"]
```

---

## Memory and Context

The context window is the agent's working memory, and it is the scarcest resource in the system. Two findings shape everything: effective attention degrades as context fills ("context rot" — models recall the middle of a long context worse than the edges), and inference cost scales with every token you keep resending. Long-horizon agents are therefore built on **context hygiene**, not maximal context.

| Layer | Mechanism | Survives |
|-------|-----------|----------|
| Working memory | The message list itself | One run, until compaction |
| Compaction summary | Model summarizes the transcript; harness restarts the loop with the summary + recent turns | Context overflow |
| File-based memory | Agent writes notes, plans, TODOs to disk (`NOTES.md`, scratch files) and re-reads them | The session — and the next one |
| Project memory | Curated instruction files (`CLAUDE.md`-style) loaded every session | The project |
| Retrieval | Search tools over a corpus or past episodes | Everything else |

Practical defaults:

- **Compaction** preserves decisions, constraints, file paths, and learned gotchas; it discards raw tool output. Trigger it at a threshold (e.g., 80% of the window), not at overflow.
- **The filesystem is the agent's external memory.** An agent that maintains its own `plan.md` and checks items off recovers from compaction and interruption almost for free.
- **Just-in-time retrieval beats preloading.** Give the agent search tools (`grep`, semantic search) and let it pull what it needs, instead of stuffing everything that might be relevant into the prompt. See [Agent Context Engineering](../18-compound-engineering/03-agent-context-engineering.md).
- Vector-store "agent memory" (embed every message, retrieve by similarity) is rarely the right first tool — explicit files the agent deliberately writes and reads are more debuggable and more faithful.

---

## Verification: The Half of the Loop That Matters

Agents shine on tasks where **checking an answer is cheaper than producing it** — code with a test suite, data transformations with invariants, UI changes you can screenshot. The harness should make ground truth available:

```mermaid
graph LR
    ACT["Act<br/>(edit code)"] --> VERIFY["Verify<br/>(run tests, typecheck, lint)"]
    VERIFY -->|fail, with errors| ACT
    VERIFY -->|pass| DONE["Done — claim with evidence"]
```

- Prefer **objective verifiers** (exit codes, diffs, pixel comparisons) over the model grading itself; self-evaluation without ground truth inflates success rates.
- Make verification *cheap and incremental*: a fast targeted test the agent can run after every edit outperforms a 20-minute suite it runs once.
- For tasks with no programmatic oracle, use rubric-based LLM-as-judge as a weak signal and route low-confidence results to a human.
- A task with no verification signal at all is a poor fit for autonomy — keep a human in the loop.

This is also the honest framing for reliability: per-step success compounds. A 98%-per-step agent finishes a 30-step task ~55% of the time. Verification steps are how you stop the compounding — they convert silent drift into a visible, recoverable error.

---

## Security and Sandboxing

An agent is an untrusted-code execution problem plus a confused-deputy problem. The harness, not the model, is the security boundary.

- **Sandbox the environment.** Run tool execution in an ephemeral container or VM: project directory mounted read-write, everything else read-only or invisible; network egress through an allowlist; secrets injected per-tool, never placed in context.
- **Classify and gate actions.** Reads auto-approve; writes inside the workspace auto-approve or batch for review; anything irreversible or outward-facing (push, deploy, send email, spend money) requires explicit approval until you have eval evidence to relax it.
- **Assume prompt injection.** Any untrusted content the agent reads — web pages, issues, emails, tool outputs — may contain instructions. Pattern-matching filters do not solve this. The structural defense is to avoid the *lethal trifecta*: an agent that simultaneously (1) reads untrusted input, (2) has access to private data, and (3) can communicate externally is exfiltratable by design. Remove or gate at least one leg.
- **Provenance matters.** Mark tool results as data, not directives, in the prompt; treat "the issue comment told me to" as a bug in your harness, not the model.

```python
IRREVERSIBLE = {"deploy", "send_email", "git_push", "payment"}

async def execute_gated(tool: str, args: dict, policy: Policy) -> str:
    action_class = classify(tool, args)          # read | write | irreversible
    if action_class == "irreversible" and not policy.pre_approved(tool, args):
        approval = await request_human_approval(tool, args)
        if not approval.granted:
            return f"Denied by operator: {approval.reason}"   # the model adapts
    return await sandbox.run(tool, args)
```

---

## Evaluating Agents

You cannot improve a harness you cannot measure. Public benchmarks calibrate expectations — SWE-bench Verified (real GitHub issues), Terminal-Bench (terminal tasks), τ-bench (tool use under policy constraints, with simulated users), OSWorld (computer use), GAIA (tool-augmented reasoning) — but your product needs its own eval set: 50–200 real tasks with programmatic graders, run on every harness change. Track *task success*, *cost per solved task*, *turns to completion*, and *unsafe-action rate*, not just model-level scores. Pass@k vs pass^k matters for agents: a task solved 1-in-8 runs is a very different product than one solved 8-in-8.

---

## Best Practices

```
1. SIMPLEST THING FIRST
   Single call → workflow → agent. Earn each step up in autonomy
   with evidence the simpler form fails.

2. DESIGN THE ENVIRONMENT, NOT JUST THE PROMPT
   Fast tests, clear errors, inspectable state. Agents are only as
   good as their feedback loops.

3. SMALL ORTHOGONAL TOOL SURFACE
   Bash + files + code execution, plus structured tools for the
   irreversible stuff. Consolidate aggressively.

4. CONTEXT HYGIENE OVER CONTEXT SIZE
   Stable prompt prefix (cache), just-in-time retrieval, compaction,
   files as memory.

5. VERIFY WITH GROUND TRUTH
   Tests, typecheckers, screenshots. The model claims; the harness
   confirms.

6. SANDBOX BY DEFAULT, GATE THE IRREVERSIBLE
   Containerized execution, egress allowlists, approval for
   outward-facing actions. Avoid the lethal trifecta.

7. BOUND EVERYTHING
   Max turns, token budgets, wall-clock timeouts, spend limits.
   Autonomy without budgets is an incident report.

8. INSTRUMENT EVERY TURN
   Traces with tokens, tools, cache hits, interventions. Evals on
   every harness change.
```

---

## References

- [Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — Anthropic's workflow/agent taxonomy
- [Effective Context Engineering for AI Agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — Anthropic
- [Writing Effective Tools for Agents](https://www.anthropic.com/engineering/writing-tools-for-agents) — Anthropic
- [Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) — scripts instead of chained tool calls
- [Model Context Protocol](https://modelcontextprotocol.io/) — specification and SDKs
- [The Lethal Trifecta for AI Agents](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) — Simon Willison
- [SWE-bench](https://www.swebench.com/) / [τ-bench](https://arxiv.org/abs/2406.12045) / [GAIA](https://arxiv.org/abs/2311.12983) — agent benchmarks
- [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://research.trychroma.com/context-rot) — Chroma research
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — the 2022 pattern that native tool calling productized

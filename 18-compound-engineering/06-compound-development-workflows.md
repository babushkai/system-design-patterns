# Compound Development Workflows

## TL;DR

Parallel agent execution requires **explicit decomposition**, **isolation**, and **integration** — the same distributed systems problems applied to development itself. A single engineer dispatching multiple AI agents across worktrees is a coordination problem. The dispatcher pattern, git worktree isolation, and structured integration protocols transform chaotic multi-agent sessions into predictable, scalable workflows. The key insight: **the human is the orchestrator, not the implementer**.

---

## The Dispatcher Pattern

### One Human, N Agents

The dispatcher pattern inverts the traditional development model. Instead of one engineer working serially on tasks, one engineer coordinates multiple agents working in parallel — each in its own isolated context.

```mermaid
flowchart TD
    HUMAN["Human Dispatcher"]

    HUMAN -->|"Task 1: Auth middleware"| AGENT_A["Agent A<br/>worktree-a"]
    HUMAN -->|"Task 2: Database migration"| AGENT_B["Agent B<br/>worktree-b"]
    HUMAN -->|"Task 3: API endpoint"| AGENT_C["Agent C<br/>worktree-c"]
    HUMAN -->|"Task 4: Test generation"| AGENT_D["Agent D<br/>worktree-d"]

    AGENT_A -->|"PR #101"| REVIEW["Integration Review"]
    AGENT_B -->|"PR #102"| REVIEW
    AGENT_C -->|"PR #103"| REVIEW
    AGENT_D -->|"PR #104"| REVIEW

    REVIEW --> MERGE["Merge to Main"]

    style HUMAN fill:#4caf50,stroke:#2e7d32,color:#fff
    style REVIEW fill:#ff9800,stroke:#e65100,color:#000
```

### Dispatcher Responsibilities

| Phase | Action | Why |
|-------|--------|-----|
| **Intake** | Receive feature request or bug report | Understand the full scope before splitting |
| **Decompose** | Break into independent, parallelizable units | Minimize inter-task dependencies |
| **Route** | Assign each unit to an agent with appropriate context | Context determines output quality |
| **Gate** | Review plans before implementation begins | Cheapest point to catch misalignment |
| **Monitor** | Check progress, unblock stuck agents | Agents fail silently on ambiguity |
| **Integrate** | Review outputs, resolve conflicts, merge | Only the dispatcher has full-system context |

### Real Session Flow

```mermaid
sequenceDiagram
    participant H as Human Dispatcher
    participant A1 as Agent 1 (worktree-auth)
    participant A2 as Agent 2 (worktree-db)
    participant A3 as Agent 3 (worktree-api)
    participant CI as CI Pipeline

    H->>A1: "Add rate limiting to /api/login"
    H->>A2: "Add last_login_at column, write migration"
    H->>A3: "Add GET /api/users/:id/activity endpoint"

    Note over H: Dispatcher reviews plans in parallel

    A1-->>H: Plan: middleware approach, Redis counter
    A2-->>H: Plan: migration + backfill script
    A3-->>H: Plan: endpoint with pagination

    H->>A1: Plan approved, implement
    H->>A2: Plan approved, implement
    H->>A3: Plan approved, but use cursor pagination not offset

    Note over A1,A3: Agents implement in parallel

    A1-->>H: PR #101 ready
    A2-->>H: PR #102 ready
    A3-->>H: PR #103 ready

    H->>CI: Merge PR #102 (no deps)
    CI-->>H: ✓ Passed

    H->>CI: Merge PR #101 (no deps)
    CI-->>H: ✓ Passed

    H->>A3: Rebase on main (picks up migration + rate limit)
    A3-->>H: Rebased, tests pass
    H->>CI: Merge PR #103
    CI-->>H: ✓ Passed
```

### Anti-Pattern: Agent-as-Orchestrator

```
❌ Ask one agent to "build the whole feature"
   → Agent works serially (slow)
   → Agent loses context mid-way through large tasks
   → No isolation between components
   → Single point of failure

✅ Human decomposes, agents implement in parallel
   → Parallel execution (fast)
   → Each agent has focused context
   → Failures are isolated
   → Human maintains system-level coherence
```

---

## Work Decomposition for Parallelism

### Decomposition Levels

The granularity of decomposition determines the maximum achievable parallelism and the coordination overhead.

| Level | Granularity | Max Parallelism | Coordination Overhead | Best For |
|-------|-------------|----------------|----------------------|----------|
| **File-level** | Each agent works on different files | High | Low | Independent modules |
| **Module-level** | Each agent owns a module boundary | Medium-High | Medium | Feature development |
| **Feature-flag** | Each agent builds behind a flag | Medium | Low | Incremental delivery |
| **Layer-level** | Frontend / backend / infra split | Medium | Medium-High | Full-stack features |
| **Phase-level** | Design → implement → test | Low | Low | Sequential dependencies |

### Dependency Graph Determines Parallelism

```mermaid
graph TD
    subgraph PARALLEL["Can Run in Parallel"]
        A["Database migration"]
        B["API types/interfaces"]
        C["UI component shell"]
    end

    subgraph SEQUENTIAL["Must Wait"]
        D["API implementation<br/>(depends on A, B)"]
        E["UI integration<br/>(depends on B, C)"]
    end

    subgraph FINAL["Final"]
        F["Integration tests<br/>(depends on D, E)"]
    end

    A --> D
    B --> D
    B --> E
    C --> E
    D --> F
    E --> F

    style PARALLEL fill:#e8f5e9,stroke:#2e7d32
    style SEQUENTIAL fill:#fff3e0,stroke:#e65100
    style FINAL fill:#fce4ec,stroke:#b71c1c
```

### Decomposition Checklist

Before assigning tasks to parallel agents, verify:

```markdown
- [ ] Each task can be completed without output from another task
- [ ] Each task operates on distinct files (no overlapping edits)
- [ ] Shared interfaces are defined upfront and provided to all agents
- [ ] Each task has a clear acceptance criteria
- [ ] Merge order is determined (which PR lands first?)
- [ ] Integration points are identified (where will outputs connect?)
```

### Decomposition by Example: User Activity Dashboard

| Agent | Scope | Files | Dependencies |
|-------|-------|-------|-------------|
| A | Database layer: migration, model, repository | `src/db/`, `src/models/` | None |
| B | API layer: endpoint, middleware | `src/handlers/`, `src/middleware/` | A (types) |
| C | Frontend: timeline component, activity hook | `src/components/`, `src/hooks/` | B (API contract) |
| D | Testing: integration, component, E2E | `tests/` | A, B, C |

**Shared contract:** `ActivityEvent` interface provided to all agents before start. **Merge order:** A, B, C, D.

### Anti-Pattern: Circular Dependencies

```
❌ Agent A needs Agent B's output, Agent B needs Agent A's output
   → Deadlock: neither can proceed
   → Root cause: insufficient upfront interface design

✅ Extract shared interfaces FIRST, then parallelize
   → Define types and contracts before any agent starts
   → Each agent implements against the interface, not the other agent's code
```

---

## Git Worktree Isolation

### Why Worktrees

Each agent needs its own working directory to avoid file conflicts. Git worktrees [1] provide this without duplicating the repository.

```
# Repository structure with worktrees
project/
├── .git/                    # Shared git database
├── src/                     # Main working tree (main branch)
├── ...
└── (worktrees created outside or inside as needed)

../project-worktrees/
├── worktree-auth/           # Agent A: feature/auth-rate-limit
├── worktree-db/             # Agent B: feature/db-migration
├── worktree-api/            # Agent C: feature/api-activity
└── worktree-tests/          # Agent D: feature/activity-tests
```

### Worktree Commands

```bash
# Setup: Create worktrees for parallel agents
cd /path/to/project

# Create branches from latest main
git fetch origin
git checkout main && git pull

# Create worktrees — each gets its own directory and branch
git worktree add ../project-worktrees/worktree-auth -b feature/auth-rate-limit
git worktree add ../project-worktrees/worktree-db -b feature/db-migration
git worktree add ../project-worktrees/worktree-api -b feature/api-activity
git worktree add ../project-worktrees/worktree-tests -b feature/activity-tests

# List active worktrees
git worktree list
# /path/to/project                    abc1234 [main]
# /path/to/project-worktrees/worktree-auth   def5678 [feature/auth-rate-limit]
# /path/to/project-worktrees/worktree-db     def5678 [feature/db-migration]
# /path/to/project-worktrees/worktree-api    def5678 [feature/api-activity]
# /path/to/project-worktrees/worktree-tests  def5678 [feature/activity-tests]
```

### Agent Assignment

```bash
# Start Agent A in the auth worktree
cd ../project-worktrees/worktree-auth
claude  # Agent A starts with focused context

# Start Agent B in the db worktree (separate terminal)
cd ../project-worktrees/worktree-db
claude  # Agent B starts with focused context

# Start Agent C in the api worktree (separate terminal)
cd ../project-worktrees/worktree-api
claude  # Agent C starts with focused context
```

### Merge Strategies

| Strategy | When to Use | Command |
|----------|-------------|---------|
| **Sequential merge** | Tasks have dependency order | Merge A, rebase B on main, merge B, ... |
| **Integration branch** | All tasks must integrate before main | Create `integration/feature-x`, merge all into it, then PR to main |
| **Feature flag merge** | Tasks are independent, behind flags | Merge all directly to main behind flags |

#### Sequential Merge

```bash
git checkout main
git merge --no-ff feature/db-migration           # no deps, merge first
git checkout feature/api-activity && git rebase main && git checkout main
git merge --no-ff feature/api-activity            # depends on db
git merge --no-ff feature/auth-rate-limit         # independent
git checkout feature/activity-tests && git rebase main && git checkout main
git merge --no-ff feature/activity-tests          # depends on all
```

#### Integration Branch

```bash
git checkout -b integration/user-activity main
git merge --no-ff feature/db-migration
git merge --no-ff feature/auth-rate-limit
git merge --no-ff feature/api-activity
git merge --no-ff feature/activity-tests
npm test
gh pr create --base main --head integration/user-activity \
  --title "feat: user activity dashboard" \
  --body "Integration of parallel agent work"
```

### Cleanup

```bash
git worktree remove ../project-worktrees/worktree-auth
git worktree remove ../project-worktrees/worktree-db
git worktree remove ../project-worktrees/worktree-api
git worktree remove ../project-worktrees/worktree-tests
git worktree prune
git branch -d feature/auth-rate-limit feature/db-migration \
    feature/api-activity feature/activity-tests
```

### Current Parallel Agent Tooling

The tooling landscape for parallel agent workflows changes quickly. Instead of anchoring the design to one product, evaluate tools by the orchestration capabilities they provide.

| Capability | What it does | Why it matters |
|------------|--------------|----------------|
| **Worktree management** | Creates isolated directories and branches per agent | Prevents file-level collisions and git index corruption |
| **Sandboxing** | Runs each agent in a constrained filesystem, network, or container boundary | Prevents dependency installs and shell commands from contaminating other sessions |
| **Approval policy** | Prompts or blocks side-effecting commands | Keeps pushes, deletes, deploys, and network calls behind human intent |
| **Subagent spawning** | Runs bounded agents in parallel for exploration, testing, or review | Reduces wall-clock time without polluting the main context |
| **Diff dashboard** | Shows all agent patches, logs, and test results in one place | Moves the bottleneck from "finding output" to "reviewing output" |
| **Cost and token tracking** | Attributes spend by task, model, and agent | Lets agent usage be managed like cloud infrastructure |
| **Connector / MCP support** | Exposes structured tools for GitHub, Linear, browser, docs, deploys | Reduces brittle shell scripting and improves auditability |
| **Handoff state** | Preserves task summary, decisions, and remaining work | Makes session restarts and agent-to-agent transfers reliable |

Examples of this pattern include local CLI agents coordinated with `git worktree`, IDE agents with inline diff review, cloud/app agents with managed sandboxes, and explicit subagent workflows in tools such as Codex and Claude Code [2][3][4]. The product names matter less than whether the runtime gives each agent a bounded task, isolated workspace, observable trace, and concrete verification step.

**The core insight:** parallelism is not the hard part. Isolation is. Creating N agents is trivial. Ensuring they do not interfere with each other's files, dependencies, git state, approvals, or context windows is the real challenge. Worktrees solve file isolation; sandboxes solve dependency and process isolation; the review and merge phase remains the sequential bottleneck.

**Skills as a cross-tool convention:** reusable task instructions are becoming portable across agent tools. Whether the mechanism is Claude Code skills/commands, Codex skills/plugins, IDE rules, or MCP prompts, the pattern is the same: package repeatable behavior outside the conversation so every agent starts with the same operational knowledge [5][6].

### Subagent Orchestration Decision Matrix

Subagents are a context-management tool before they are a speed tool. They keep noisy exploration, logs, and partial findings out of the main session. Use them aggressively for read-heavy work and conservatively for write-heavy work.

| Work type | Parallel subagents? | Why | Integration rule |
|-----------|---------------------|-----|------------------|
| Codebase exploration | Yes | Each agent can inspect a different subsystem and return a concise summary | Parent keeps decisions; subagents return evidence and file references |
| Test/log triage | Yes | Failures can be partitioned by suite, service, or stack trace | Parent merges root-cause findings before assigning fixes |
| Documentation conversion | Yes | File-level tasks are independent | Assign disjoint file lists and run one final build |
| Security review | Yes, as reviewers | Multiple agents catch different risk classes | Human remains final gate; require severity and file references |
| Feature implementation | Sometimes | Works only when interfaces and file ownership are clear | Predefine contracts, owners, and merge order |
| Shared schema or migration work | Usually no | Small conflicts have large blast radius | One owner writes the contract; others consume it |
| Architecture decisions | No | Requires coherent trade-off judgment | Use agents for options research, not final decision-making |
| Production deploy / DNS / secrets | No | External side effects need explicit human control | Agent can prepare commands; human approves execution |

The safest pattern is **fan-out / summarize / decide / fan-out**:

```mermaid
sequenceDiagram
    participant H as Human / Main Agent
    participant E1 as Explorer 1
    participant E2 as Explorer 2
    participant E3 as Explorer 3
    participant W1 as Worker 1
    participant W2 as Worker 2

    H->>E1: Inspect auth subsystem
    H->>E2: Inspect data model
    H->>E3: Inspect tests and CI
    E1-->>H: Summary + file refs
    E2-->>H: Summary + file refs
    E3-->>H: Summary + file refs
    H->>H: Decide architecture and contracts
    H->>W1: Implement owned slice
    H->>W2: Implement owned slice
    W1-->>H: Patch + verification
    W2-->>H: Patch + verification
    H->>H: Integrate and run final checks
```

Do not let subagents recursively become the project manager. The main dispatcher owns task decomposition, acceptance criteria, merge order, and final verification.

### The Isolation Hierarchy

Not all isolation is equal. The right level depends on the task's risk profile, the team's infrastructure, and the cost budget.

| Level | Mechanism | File Isolation | Dependency Isolation | Cost | Use Case |
|-------|-----------|---------------|---------------------|------|----------|
| **Level 0** | Same directory, same branch | None | None | Free | Single agent, sequential work |
| **Level 1** | Same repo, different branches | Partial (uncommitted changes conflict) | None | Free | Low-parallelism, careful coordination |
| **Level 2** | Git worktrees | Full (separate working directories) | Shared (same node_modules, etc.) | Free | Most local parallel workflows |
| **Level 3** | Container / app sandboxes | Full | Full (each container has own deps) | Variable | CI/CD agents, dependency installs, untrusted code |
| **Level 4** | Cloud VMs | Complete | Complete | Variable | Maximum isolation, compliance requirements |

**Level 0** is where most developers start — one agent, one directory. Agents will conflict the moment you try to run two in the same working tree. File locks, partial writes, and git index corruption are common.

**Level 1** seems like it should work, but uncommitted changes in one branch are visible to `git stash` and `git checkout` operations in another. Merge conflicts are likely when branches diverge significantly.

**Level 2 (worktrees)** is the sweet spot for most teams. Each agent gets a fully independent working directory with its own checked-out branch, but all worktrees share the same `.git` database. No repository duplication, no extra disk cost beyond the working files, and full git history available everywhere.

**Level 3 and 4** add dependency and OS-level isolation. These matter when agents install packages, run build tools, or execute untrusted code. A rogue `npm install` in one container cannot corrupt another agent's `node_modules`.

**Decision heuristic:** Use Level 2 (worktrees) for local development workflows. Use Level 3 (containers) for CI/CD agents and when agents install dependencies. Use Level 4 (cloud VMs) only when compliance or security policies require complete separation.

---

## Background Task Patterns

### Fire-and-Review

Some tasks are well-suited to background execution: the agent works independently while the dispatcher focuses elsewhere, returning later to review the output.

```mermaid
flowchart LR
    subgraph FOREGROUND["Foreground (Human Focus)"]
        DESIGN["Design new feature"]
        REVIEW["Review agent outputs"]
    end

    subgraph BACKGROUND["Background (Agent Tasks)"]
        TEST_GEN["Generate test cases<br/>for module X"]
        DOC_UPDATE["Update API docs<br/>from code changes"]
        REFACTOR["Extract shared<br/>utility functions"]
        LINT_FIX["Fix all lint<br/>warnings in /src"]
    end

    DESIGN -.->|"Dispatches"| TEST_GEN
    DESIGN -.->|"Dispatches"| DOC_UPDATE
    DESIGN -.->|"Dispatches"| REFACTOR
    DESIGN -.->|"Dispatches"| LINT_FIX

    TEST_GEN -->|"PR ready"| REVIEW
    DOC_UPDATE -->|"PR ready"| REVIEW
    REFACTOR -->|"PR ready"| REVIEW
    LINT_FIX -->|"PR ready"| REVIEW

    style FOREGROUND fill:#e8f5e9,stroke:#2e7d32
    style BACKGROUND fill:#e3f2fd,stroke:#1565c0
```

### Good Background Tasks

| Task Type | Why It Works in Background | Review Effort |
|-----------|---------------------------|---------------|
| Test generation for existing code | Well-defined scope, existing code as spec | Medium (verify coverage) |
| Documentation updates | Low risk, easily verified | Low (accuracy check) |
| Lint/format fixes | Mechanical, tools verify correctness | Low (skim diff) |
| Type annotation additions | Compiler verifies correctness | Low (type check passes) |
| Dependency updates | CI verifies compatibility | Medium (changelog review) |
| Code migration (syntax upgrades) | Pattern-based, testable | Medium (verify behavior) |

### Poor Background Tasks

| Task Type | Why It Fails |
|-----------|-------------|
| New feature implementation | Needs iterative feedback — agent goes in wrong direction |
| Architecture decisions | Needs human judgment — agent picks "popular" over "appropriate" |
| Security-sensitive changes | Needs expert review during design |
| Cross-module refactors | Needs whole-system understanding |

### Background Task Template

```markdown
## Background Task: [Title]
### Scope: [Exact files/directories]
### Instructions: [Step-by-step]
### Constraints: Do NOT modify files outside scope. No new dependencies.
### Acceptance: All existing tests pass + [task-specific criteria]
### When Done: Create PR with label "background-task"
```

---

## Parallel Agent Coordination

### Shared State Hazards

When multiple agents work in parallel, shared state creates the same hazards as concurrent programming.

| Hazard | Example | Mitigation |
|--------|---------|------------|
| **Write conflict** | Two agents modify the same file | File-level ownership per agent |
| **Schema conflict** | Two agents add migrations with same sequence number | Pre-assign migration numbers |
| **Import conflict** | Two agents add different exports to index.ts | Merge via integration branch |
| **Convention drift** | Agents adopt different naming conventions | Provide style guide in context |
| **Dependency conflict** | Agent A adds lodash, Agent B adds ramda | Pre-approve dependencies |
| **Type conflict** | Two agents define overlapping types | Define shared types upfront |

### Conflict Detection

```bash
# Check all pairs of parallel branches for conflicts before integration
BRANCHES=(feature/auth feature/db feature/api feature/tests)
for i in "${BRANCHES[@]}"; do
    for j in "${BRANCHES[@]}"; do
        if [ "$i" != "$j" ]; then
            git checkout "$i" 2>/dev/null
            if ! git merge --no-commit --no-ff "$j" 2>/dev/null; then
                echo "CONFLICT: $i ↔ $j"
            fi
            git merge --abort 2>/dev/null
        fi
    done
done
```

### Spec-First Coordination Pattern

The most effective coordination strategy: define all interfaces before any agent starts implementing.

```mermaid
flowchart TD
    SPEC["1. Human Defines Interfaces"]
    SHARE["2. Share Spec with All Agents"]
    PARALLEL["3. Agents Implement in Parallel"]
    INTEGRATE["4. Integrate Against Interfaces"]

    SPEC --> SHARE
    SHARE --> PARALLEL
    PARALLEL --> INTEGRATE

    subgraph SPEC_CONTENTS["Interface Spec Contains"]
        TYPES["Type definitions"]
        CONTRACTS["API contracts"]
        EVENTS["Event schemas"]
        NAMING["Naming conventions"]
    end

    SPEC --- SPEC_CONTENTS

    style SPEC fill:#4caf50,stroke:#2e7d32,color:#fff
```

### Shared Interface Example

```typescript
// shared-types.ts — created by human BEFORE agents start, read-only for all agents
export interface ActivityEvent {
  id: string;
  userId: string;
  type: 'login' | 'logout' | 'create' | 'update' | 'delete';
  resource: string;
  timestamp: Date;
  metadata: Record<string, unknown>;
}

export interface PaginatedResponse<T> {
  items: T[];
  cursor: string | null;
  hasMore: boolean;
}

export enum ErrorCode {
  NOT_FOUND = 'NOT_FOUND',
  INVALID_INPUT = 'INVALID_INPUT',
  UNAUTHORIZED = 'UNAUTHORIZED',
  RATE_LIMITED = 'RATE_LIMITED',
}
```

---

## The Integration Protocol

### Staged Review and Merge

```mermaid
sequenceDiagram
    participant A1 as Agent 1 PR
    participant A2 as Agent 2 PR
    participant A3 as Agent 3 PR
    participant H as Human Reviewer
    participant IB as Integration Branch
    participant CI as CI Pipeline
    participant M as Main Branch

    Note over A1,A3: All agents complete in parallel

    H->>A1: Review PR #101
    H->>A2: Review PR #102
    H->>A3: Review PR #103

    Note over H: Reviews happen in parallel

    H->>IB: Create integration/feature-x from main
    H->>IB: Merge PR #101 (db migration)
    H->>CI: Run CI on integration branch
    CI-->>H: ✓ Passed

    H->>IB: Merge PR #102 (API layer)
    H->>CI: Run CI on integration branch
    CI-->>H: ✓ Passed

    H->>A3: Rebase PR #103 on integration branch
    A3-->>H: Rebased, conflicts resolved
    H->>IB: Merge PR #103 (frontend)
    H->>CI: Run full CI + integration tests
    CI-->>H: ✓ All passed

    H->>M: PR: integration/feature-x → main
    CI-->>H: ✓ Final check passed
    H->>M: Merge
```

### Integration Checklist

```markdown
## Pre-Integration
- [ ] All individual PRs reviewed and approved
- [ ] No dependency conflicts between branches
- [ ] Shared types/interfaces match across all branches
- [ ] Migration sequence numbers don't conflict

## During Integration
- [ ] Merge in dependency order (db → api → frontend → tests)
- [ ] Run CI after each merge
- [ ] Resolve conflicts with full context (not just git's suggestion)

## Post-Integration
- [ ] Full test suite passes
- [ ] Integration/E2E tests pass
- [ ] No regressions in existing functionality
- [ ] Performance benchmarks within acceptable range
```

### Conflict Resolution Decision Table

| Conflict Type | Resolution Strategy | Who Resolves |
|---------------|-------------------|--------------|
| Import ordering | Accept either, run formatter | Automated |
| Package lock file | Regenerate from scratch | Automated |
| Same file, different sections | Manual merge | Human |
| Same function, different logic | Choose based on spec | Human |
| Type definition mismatch | Align to shared spec | Human |
| Test file overlap | Combine test suites | Human or Agent |
| Configuration overlap | Merge settings, test | Human |

---

## Workflow Patterns Catalog

### Pattern 1: Spike-and-Converge

**When:** Uncertain about approach, need to explore options.

```mermaid
flowchart TD
    PROBLEM["Problem Statement"]
    A1["Agent A: Approach 1<br/>e.g., Redis-based"]
    A2["Agent B: Approach 2<br/>e.g., In-memory"]
    A3["Agent C: Approach 3<br/>e.g., Database-based"]
    EVAL["Human Evaluates<br/>All Three Spikes"]
    CHOOSE["Select Best Approach"]
    IMPLEMENT["Full Implementation<br/>(one agent, chosen approach)"]

    PROBLEM --> A1
    PROBLEM --> A2
    PROBLEM --> A3
    A1 --> EVAL
    A2 --> EVAL
    A3 --> EVAL
    EVAL --> CHOOSE
    CHOOSE --> IMPLEMENT
```

**Time cost:** 3× agent compute for spike phase, but saves days of wrong-direction implementation.

**Decision criteria for spike evaluation:**

| Criterion | Weight | How to Evaluate |
|-----------|--------|----------------|
| Correctness | 30% | Does it handle all edge cases? |
| Simplicity | 25% | Lines of code, dependency count |
| Performance | 20% | Benchmark results |
| Maintainability | 15% | Readability, test coverage |
| Operational cost | 10% | Infrastructure requirements |

### Pattern 2: Test-First Delegation

**When:** Well-specified behavior, need reliable implementation.

```mermaid
flowchart LR
    SPEC["Spec / Requirements"]
    HUMAN["Human Writes<br/>Test Cases"]
    AGENT["Agent Implements<br/>Until Tests Pass"]
    REVIEW["Review Implementation"]

    SPEC --> HUMAN
    HUMAN --> AGENT
    AGENT --> REVIEW

    style HUMAN fill:#4caf50,stroke:#2e7d32,color:#fff
```

**Why it works:** Tests written by humans encode the spec. The agent's job is reduced to making tests pass — a well-defined, verifiable objective. This eliminates the co-generation problem entirely.

```bash
# Human writes tests first
# tests/auth/rate-limit.test.ts

# Then dispatches to agent with clear instruction:
# "Implement src/middleware/rate-limit.ts to make all tests in
#  tests/auth/rate-limit.test.ts pass. Do not modify the test file."
```

### Pattern 3: Refactor Pipeline

**When:** Large-scale refactoring across many files.

```
PHASE 1: Analysis (1 agent)
  → Identify all usage sites
  → Generate dependency graph
  → Propose refactor plan

PHASE 2: Execution (N agents, parallel)
  → Agent A: Refactor module X
  → Agent B: Refactor module Y
  → Agent C: Refactor module Z
  → Each agent has the shared interface contract

PHASE 3: Verification (1 agent)
  → Run full test suite
  → Check for orphaned imports
  → Verify no behavior changes
```

### Pattern 4: Documentation Sweep

**When:** Docs have drifted from code, need bulk updates.

```
ASSIGNMENT: One agent per documentation area
  → Agent A: API reference docs (reads routes, generates OpenAPI)
  → Agent B: Architecture docs (reads code, updates diagrams)
  → Agent C: README and quickstart (reads setup, verifies steps)
  → Agent D: Inline code comments (reads complex functions, adds JSDoc)

CONSTRAINT: Agents read code, write docs. Never modify source code.
```

### Pattern 5: Review-in-Parallel

**When:** Multiple PRs queued, need faster review throughput.

```mermaid
flowchart LR
    PR1["PR #201"]
    PR2["PR #202"]
    PR3["PR #203"]

    A1["Agent A:<br/>Review #201<br/>Check logic, tests"]
    A2["Agent B:<br/>Review #202<br/>Check logic, tests"]
    A3["Agent C:<br/>Review #203<br/>Check logic, tests"]

    HUMAN["Human Reviews<br/>Agent Findings"]

    PR1 --> A1
    PR2 --> A2
    PR3 --> A3
    A1 --> HUMAN
    A2 --> HUMAN
    A3 --> HUMAN

    style HUMAN fill:#4caf50,stroke:#2e7d32,color:#fff
```

**Agent review prompt:** Check for logic errors (edge cases, error handling), security issues, test coverage gaps, API contract violations, and performance concerns. Output as BLOCKING / WARNING / NOTE.

### Pattern 6: Security-Concurrent

**When:** Implementing features that touch security boundaries.

```
PARALLEL EXECUTION:
  Agent A: Implement the feature (functional)
  Agent B: Write security test cases (adversarial)
  Agent C: Run threat model analysis (analytical)

INTEGRATION:
  1. Agent B's security tests must pass against Agent A's implementation
  2. Agent C's threat model findings must be addressed
  3. Human reviews the combined output with security focus
```

### Pattern Summary

| Pattern | Parallelism | Risk Level | Best For |
|---------|------------|-----------|----------|
| Spike-and-Converge | Exploration | Low (throwaway) | Architecture decisions |
| Test-First Delegation | Low (sequential) | Low (tests are oracle) | Well-specified features |
| Refactor Pipeline | High (execution phase) | Medium | Large-scale changes |
| Documentation Sweep | High | Low | Bulk doc updates |
| Review-in-Parallel | High | Low | PR queue backlog |
| Security-Concurrent | Medium | Medium-High | Security-sensitive features |

---

## Cost Management

### API Spend as Engineering Concern

AI agent usage has a direct cost component [8]. Unlike human engineers (fixed salary), agent costs scale with usage and must be budgeted like cloud infrastructure.

### Token Budget by Task Type

| Task Type | Estimated Tokens | Model Recommendation | Cost Estimate |
|-----------|-----------------|---------------------|---------------|
| Plan generation | 2K-5K output | Sonnet | $0.02-0.05 |
| Code implementation (small) | 5K-15K output | Sonnet | $0.05-0.15 |
| Code implementation (large) | 15K-50K output | Sonnet | $0.15-0.50 |
| Code review | 3K-10K output | Sonnet | $0.03-0.10 |
| Test generation | 10K-30K output | Sonnet | $0.10-0.30 |
| Architecture spike | 5K-20K output | Opus | $0.30-1.20 |
| Complex debugging | 10K-40K output | Opus | $0.60-2.40 |
| Documentation | 5K-15K output | Sonnet | $0.05-0.15 |
| Lint/format fixes | 2K-8K output | Haiku | $0.002-0.008 |
| Type annotations | 3K-10K output | Haiku | $0.003-0.010 |

### Model Selection Rules

| Signal | Model | Rationale |
|--------|-------|-----------|
| Architecture reasoning, complex debugging, security review | Opus | Requires deep judgment |
| Feature implementation, code review, test generation | Sonnet | Default for most work |
| Formatting, linting, type annotations, simple refactors | Haiku | Mechanical, low-judgment |

### Cost Tracking

Track token usage per task in an append-only log (e.g., `.claude/cost-log.jsonl`). Aggregate weekly to identify expensive task categories and model mismatches.

### Anti-Pattern: Using Opus for Everything

```
❌ Use Opus for all tasks
   → $50/day for work that could cost $5/day
   → No quality improvement for mechanical tasks
   → Budget exhausted before sprint ends

✅ Match model to task complexity
   → Opus for architecture and debugging ($2-5/task)
   → Sonnet for implementation ($0.10-0.50/task)
   → Haiku for mechanical work ($0.01/task)
   → 10× cost reduction with equivalent quality
```

---

## Session Management

### The Context Window Problem

Agent sessions have finite context windows. Long sessions degrade in quality as the context fills with stale information. Effective session management preserves continuity across context boundaries.

### Resume Patterns

| Pattern | When to Use | What to Include |
|---------|-------------|-----------------|
| **Checkpoint summary** | Mid-task pause | Completed items, in-progress items, next steps, key decisions, files modified |
| **Context summarization** | New session resume | Project context, what exists, what needs doing, conventions |
| **Handoff prompt** | Agent-to-agent transfer | Prior agent output, remaining task, interface contract, constraints |

#### Checkpoint Summary Template

```markdown
## Session Checkpoint — 2026-03-15 14:30

### Completed
- [x] Created database migration for activity_events table
- [x] Implemented ActivityEvent model with types

### In Progress
- [ ] API endpoint (handler skeleton exists)

### Next Steps
1. Complete request validation in handler
2. Add cursor-based pagination

### Key Decisions
- Cursor pagination (not offset) — reviewer feedback
- Timestamp-based cursor using created_at
```

#### Handoff Prompt Template

```markdown
## Handoff: Agent A → Agent B

### Agent A's Output
- PR #102 (merged): Database migration + model + repository

### Agent B's Task
Implement GET /api/users/:id/activity endpoint.

### Constraints
- Do NOT modify any files Agent A created
- Follow existing handler patterns in src/handlers/
- Use cursor-based pagination
```

### Session Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Fresh: New task
    Fresh --> Active: Agent starts work
    Active --> Checkpoint: Context getting large
    Checkpoint --> Resume: New session with summary
    Resume --> Active: Agent continues
    Active --> Handoff: Transferring to another agent
    Handoff --> Fresh: New agent starts
    Active --> Complete: Task finished
    Complete --> [*]

    note right of Checkpoint
        Save progress summary,
        file list, decisions made
    end note

    note right of Handoff
        Include completed work,
        remaining tasks,
        interface contracts
    end note
```

### Anti-Pattern: Infinite Sessions

```
❌ Keep one agent session running for hours
   → Context window fills with irrelevant history
   → Agent starts contradicting its earlier decisions
   → Quality degrades progressively
   → Errors compound (agent forgets constraints)

✅ Checkpoint every 30-45 minutes of active work
   → Fresh context for each phase
   → Summary preserves decisions without noise
   → Consistent quality throughout the task
   → Clear audit trail of progress
```

---

## Real-World Example

### End-to-End: Converting a Codebase to Use Mermaid Diagrams

**Scenario:** A documentation repository has 47 ASCII diagrams across 4 directories that need Mermaid conversion. Each diagram is independent — ideal for parallel agents.

```bash
# Step 1: Decompose (Human, 10 min)
grep -rl "┌\|├\|└\|│\|───" docs/ | wc -l  # 47 files
# Group A: docs/architecture/ (12), Group B: docs/api/ (10)
# Group C: docs/guides/ (13),      Group D: docs/reference/ (12)

# Step 2: Create worktrees (Human, 2 min)
git fetch origin && git checkout main && git pull
git worktree add ../wt/arch -b refactor/mermaid-architecture
git worktree add ../wt/api -b refactor/mermaid-api
git worktree add ../wt/guides -b refactor/mermaid-guides
git worktree add ../wt/ref -b refactor/mermaid-reference

# Step 3: Dispatch agents — one per worktree (Human, 5 min)
# Each agent gets: file list, conversion instructions, constraints

# Step 4: Integrate (Human, 20 min)
git checkout -b refactor/mermaid-all main
git merge --no-ff refactor/mermaid-architecture
git merge --no-ff refactor/mermaid-api
git merge --no-ff refactor/mermaid-guides
git merge --no-ff refactor/mermaid-reference
npm run docs:build  # verify rendering

# Step 5: Cleanup
git worktree remove ../wt/arch ../wt/api ../wt/guides ../wt/ref
git worktree prune
```

### Results

| Metric | Serial (1 agent) | Parallel (4 agents) |
|--------|-------------------|---------------------|
| Wall clock time | ~2 hours | ~35 minutes |
| Human time | ~30 min (review) | ~40 min (dispatch + review) |
| Agent compute cost | ~$2.00 | ~$2.00 (same total work) |
| Merge conflicts | 0 | 0 (file-level isolation) |
| Conversion errors | ~3 | ~3 (same error rate) |

**Key insight:** Parallel execution reduces wall clock time by ~70% with only marginally more human coordination time. The total compute cost is identical — parallelism is free when tasks are independent.

---

## Key Takeaways

1. **The human is the orchestrator, not the implementer.** The dispatcher pattern inverts the development model: one human coordinates N agents. The human's job is decomposition, routing, gating, and integration — the highest-leverage activities.

2. **Decomposition quality determines parallelism.** Independent tasks can run fully in parallel. Dependent tasks must be sequenced. Investing time in clean decomposition pays dividends in execution speed.

3. **Git worktrees provide free isolation.** Each agent gets its own working directory and branch without duplicating the repository. This eliminates file-level conflicts during parallel execution.

4. **Define interfaces before dispatching agents.** The spec-first coordination pattern — defining shared types and contracts before any agent starts — prevents the most common integration failures.

5. **Match the model and runtime to the task.** Use faster/cheaper models for mechanical work, stronger models for architecture and security, local runtimes for environment-specific tasks, and sandboxed/cloud runtimes for risky execution.

6. **Session management is a first-class concern.** Checkpoint summaries, context compression, and handoff prompts maintain quality across context window boundaries. Avoid infinite sessions.

7. **Background tasks are fire-and-review.** Test generation, documentation updates, lint fixes, and type annotations are excellent background tasks. New features and architecture decisions are not.

8. **Integration is where parallel work becomes sequential.** Use integration branches, merge in dependency order, and run CI after each merge. The integration protocol is the critical path.

9. **Cost management is engineering management.** Track token usage by task type. Budget agent spend like cloud infrastructure. The cheapest token is the one you don't send.

10. **Parallel execution time savings are real.** Independent tasks achieve near-linear speedup with parallel agents. The total compute cost is unchanged — you pay the same tokens, just faster.

---

## References

1. [Git — git-worktree Documentation](https://git-scm.com/docs/git-worktree)
2. [OpenAI — Codex Subagents](https://developers.openai.com/codex/concepts/subagents)
3. [OpenAI — Codex Sandbox](https://developers.openai.com/codex/concepts/sandboxing)
4. [Anthropic — Claude Code Overview](https://code.claude.com/docs/en/overview)
5. [Anthropic — Claude Code Best Practices](https://code.claude.com/docs/en/best-practices)
6. [OpenAI — Custom Instructions with AGENTS.md](https://developers.openai.com/codex/guides/agents-md)
7. [Nx Blog — Using Git Worktrees for Parallel Agent Development](https://nx.dev/blog/using-git-worktrees-for-parallel-agent-development)

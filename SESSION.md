# SESSION.md

- `create_react_agent` is marked deprecated in the latest LangGraph reference
  docs (v1.1.6). The recommended path forward is building a `StateGraph`
  directly. Keeping it for now since it works and the migration is
  straightforward.
- No checkpointer is wired into `main.py` yet, so conversation memory does not
  persist across turns within the REPL. ~~Pass `InMemorySaver()` to
  `build_graph()` to enable it.~~ **Fixed**: `relay/cli/repl.py` now passes
  `MemorySaver()` to `build_graph()`.
- LangSmith tracing is documented in `.env.example` but not tested.
- The Relay system prompt currently over-enforces filesystem discovery
  (`ALWAYS use ls/glob_files/grep_files`) even for high-level questions like
  project summaries, which causes avoidable first-turn tool churn.
- `AgentContext.user_memory` is exposed in the prompt template, but the CLI
  streaming path constructs `AgentContext()` with the default empty string
  instead of loading any persisted user/project memory.
  ~~**Fixed**~~: `stream_response` now loads `.relay/memory.md` into
  `AgentContext.user_memory`.
- The interactive CLI special-cases `/resume` in `relay/cli/session.py` because
  command dispatch is not fully async, which makes the REPL loop own too much
  workflow logic.
- Assistant streaming still writes directly from `relay/cli/streaming.py`
  instead of routing all output through a single renderer layer.
- `relay/tools/planning.py` (coordinator-safe `think` tool) is now dead code —
  the coordinator no longer uses `think` (matching langrepl's architecture).
  The file and its test (`tests/tools/test_planning.py`) can be removed.
- ~~**Phase 6 deferred**: `ApprovalMiddleware` and `SandboxMiddleware` require
  config subsystem types (`ApprovalMode`, `ToolApprovalConfig`, `ToolApprovalRule`)
  and a `SandboxBackend` abstraction that relay does not yet have. These should
  be added after the config infrastructure from Phase 4 is more mature.~~
  **Resolved**: `ApprovalMiddleware`, `SandboxMiddleware`, approval config types
  (`ApprovalMode`, `ToolApprovalConfig`, `ToolApprovalRule`), and the
  `SandboxBackend` ABC are now implemented. Sandbox backends (bubblewrap,
  seatbelt) still need OS-specific implementations.
- The `ApprovalMiddleware` and `SandboxMiddleware` are exported from
  `relay.middlewares` but are **not yet registered** in the default
  `create_react_agent` middleware stack because they require runtime
  configuration (approval mode, tool-sandbox map) that the CLI does not
  yet wire up.
- `relay/cli/session.py` and `relay/cli/streaming.py` are dead files from
  before the CLI restructuring into `core/`, `dispatchers/`, `handlers/`.
  They can be deleted once we confirm nothing else references them.
- `prompt_for_interrupt` still lives in `core/streaming.py` as a free function
  used by the interrupt/resume loop inside `stream_response`. The new
  `InterruptHandler` class duplicates the same logic. Eventually
  `stream_response` should delegate to `InterruptHandler` to remove the
  duplication.
- Relay's built-in `InMemoryRateLimiter` is request-based (`requests_per_second`)
  but the OpenAI failure mode seen in the CLI is token-per-minute. It can slow
  request bursts, but it does not estimate prompt size or prevent large-tool-
 output turns from hitting provider TPM limits.
- The hardcoded fallback agent path (`AgentFactory.create()`) still uses the
  generic prompt aliases from `relay/prompt.py`, which instruct the explorer to
  use todo-oriented tools it does not have. The config-driven path uses the
  dedicated prompt files under `relay/resources/configs/default/prompts/` and
  does not have this mismatch.
- Delegated subagents currently stream only `updates` inside
  `relay/tools/subagents/task.py`, so subagent assistant text chunks are never
  forwarded to the parent CLI. The user only sees structured tool activity plus
  the final synthesized tool result.
- `relay/cli/core/streaming.py` and `relay/cli/ui/renderer.py` reduce assistant
  output to plain text rendering and do not preserve provider-native
  `thinking`/`reasoning` blocks or merged `AIMessage` objects the way the
  langrepl reference implementation does.
- `relay/cli/core/streaming.py` only renders an update-path `AIMessage` when no
  prior assistant text has been collected for the turn, so later narration
  phases can be suppressed even outside the subagent path.

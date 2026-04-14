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
- The interactive CLI special-cases `/resume` in `relay/cli/session.py` because
  command dispatch is not fully async, which makes the REPL loop own too much
  workflow logic.
- Assistant streaming still writes directly from `relay/cli/streaming.py`
  instead of routing all output through a single renderer layer.

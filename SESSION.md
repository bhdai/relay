# SESSION.md

- `create_react_agent` is marked deprecated in the latest LangGraph reference
  docs (v1.1.6). The recommended path forward is building a `StateGraph`
  directly. Keeping it for now since it works and the migration is
  straightforward.
- No checkpointer is wired into `main.py` yet, so conversation memory does not
  persist across turns within the REPL. Pass `InMemorySaver()` to
  `build_graph()` to enable it.
- LangSmith tracing is documented in `.env.example` but not tested.

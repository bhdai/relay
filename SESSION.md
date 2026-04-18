# SESSION.md
- `langgraph.json` still points to `relay/graph.py`, which uses the hardcoded
  `AgentFactory.create()` fallback instead of the CLI's config-driven
  `Initializer` -> `AgentFactory.create_from_config(...)` path. LangGraph
  Studio can therefore diverge from `.relay` agent configs, MCP tools, and
  loaded skills.
- Packaged default agents do not declare sandbox config, so the current
  shipped runtime does not instantiate `SandboxMiddleware` even though the
  sandbox subsystem exists.
- Delegated subagents built through `relay/tools/subagents/task.py` do not
  inherit coordinator sandbox bindings or tool module maps, so sandbox policy
  is inconsistent between the parent agent and task-created child agents.
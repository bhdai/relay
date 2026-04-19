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
- (Phase 2) `relay/cli/core/streaming.py`: `prompt_for_interrupt` uses a
  module-level `_current_mode` list as a temporary shim for the removed
  `context.approval_mode` field.  The mode cycling UI is functionally
  preserved but not wired to `permission_ruleset` yet — this is Phase 5 work
  (`relay/cli/core/context.py` migration).
- (Phase 2) `relay/middlewares/approval.py` uses
  `getattr(context, "approval_mode", ApprovalMode.SEMI_ACTIVE)` as a
  fallback since `approval_mode` was removed from `AgentContext`.  The entire
  approval middleware is replaced in Phase 3 by `PermissionMiddleware`.
- (Phase 2) `tests/middlewares/test_approval.py` `_make_request` now
  simulates AGGRESSIVE bypass via `always_approve` metadata instead of
  `approval_mode`; review when Phase 3 replaces the approval middleware.
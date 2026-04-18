# Relay Sandboxing Implementation Analysis

## Executive Summary

The Relay project has a sophisticated sandboxing system that uses **bubblewrap (bwrap)** for OS-level isolation on Linux. However, there is a **critical gap** between the sandboxing infrastructure and actual command execution: the `run_command` tool (shell execution) **does not enforce working directory boundaries** and can escape the intended project scope.

---

## 1. How Sandboxing is Implemented

### Architecture Overview

The sandboxing system consists of four main components:

#### 1.1 **SandboxBackend (Abstract Base)**
**File:** `relay/sandboxes/backend.py`

- Abstract protocol that all sandbox backends must implement
- Handles subprocess lifecycle management and worker communication
- Only needs to implement two methods:
  - `build_command()`: Wrap the command in sandbox invocation
  - `validate_environment()`: Check if sandbox binary exists

Key features:
- JSON-based protocol: requests and responses pass through stdin/stdout as JSON
- Output collection: Truncates stderr/stdout to prevent memory exhaustion (10MB/1MB limits)
- Process management: Handles timeouts, process groups, and graceful shutdown
- Environment isolation: Passes controlled environment variables to subprocess

#### 1.2 **BubblewrapBackend (Linux Implementation)**
**File:** `relay/sandboxes/bubblewrap.py`

Uses **bubblewrap** (Linux-only) for namespace isolation:

**Filesystem Isolation:**
- Starts with empty `tmpfs` root filesystem
- Mounts only explicitly allowed paths as read-only (`--ro-bind`) or writable (`--bind`)
- Hides sensitive paths by mapping them to `/dev/null` (files) or empty `tmpfs` (directories)

**Namespace Isolation:**
```bash
--unshare-user     # User namespace
--unshare-pid      # Process namespace
--unshare-ipc      # IPC namespace
--unshare-uts      # UTS (hostname) namespace
--unshare-net      # Network namespace (optional)
```

**Seccomp BPF Filtering (Optional):**
Blocks dangerous syscalls that could enable sandbox escape:
- `ptrace` (process tracing/debugging)
- `AF_UNIX` sockets (unless explicitly allowed)
- Kernel module loading (`init_module`, `finit_module`)
- Cross-process memory access (`process_vm_readv`, `process_vm_writev`)
- Kernel keyring operations
- And others (28 syscalls total)

#### 1.3 **Worker Module**
**File:** `relay/sandboxes/worker.py`

Entry point: `python -m relay.sandboxes.worker`

- Runs **inside** the bwrap sandbox
- Reads JSON request from stdin
- Module allowlist: Only imports from `relay.tools.*` prefix
- Invokes tool via `ainvoke()` method
- Returns JSON response to stdout

**Security property:** Prevents sandboxed process from importing arbitrary modules.

#### 1.4 **SandboxMiddleware**
**File:** `relay/middlewares/sandbox.py`

LangChain middleware that intercepts tool calls:

- **Binding resolution:** Matches tool names to patterns (three-part format: `category:module:name`)
- **Three outcomes:**
  1. **Blocked:** No binding matched → tool execution blocked
  2. **Passthrough:** Binding matched with `backend=None` → tool runs unsandboxed
  3. **Sandboxed:** Binding matched with backend → tool runs in sandbox

Pattern matching uses fnmatch wildcards:
- `impl:terminal:*` matches all terminal tools
- `impl:file_system:read_file` matches specific file tool
- `!terminal:run_command` negates patterns

---

### 2. Sandbox Configuration

**File:** `relay/configs/sandbox.py`

Three-level configuration:

```yaml
# Profile (YAML file)
sandbox_config = SandboxConfig(
    name="default-linux",
    type=SandboxType.BUBBLEWRAP,
    os=SandboxOS.LINUX,
    filesystem=FilesystemConfig(
        read=[".", "/usr", "/lib", "/lib64"],    # Read-only paths
        write=["."],                              # Writable paths
        hidden=["~/.ssh", "~/.aws", ".env"]      # Hidden paths
    ),
    network=NetworkConfig(
        remote=["*"],                             # "*" = all, [] = blocked
        local=[]
    )
)

# Agent-level binding
AgentSandboxConfig(
    enabled=True,
    profiles=[
        SandboxProfileBinding(
            patterns=["impl:terminal:*"],
            sandbox=sandbox_config
        ),
        SandboxProfileBinding(
            patterns=["internal:*:*"],
            sandbox=None  # Passthrough (unsandboxed)
        )
    ]
)
```

Key points:
- `"."` represents agent's `working_dir` 
- Paths are resolved at runtime against `working_dir`
- Glob patterns supported: `~/.ssh`, `/tmp/*.pem`
- Network is binary: either all allowed or all blocked

---

## 3. Why Agents Can Run Commands Outside Project Scope

### The Critical Gap

**Status:** Session.md (lines 7-11) already identifies this issue.

The problem has **two parts**:

#### Part 1: `run_command` Does Not Enforce Working Directory

**File:** `relay/tools/impl/terminal.py`, lines 78-82

```python
@tool
async def run_command(command: str) -> str:
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )  # NOTE: No cwd parameter!
    stdout_bytes, stderr_bytes = await proc.communicate()
```

The `asyncio.create_subprocess_shell()` call:
- **Does NOT pass `cwd`** parameter
- **Does NOT restrict** the shell to `working_dir`
- Inherits the parent process's working directory (wherever relay was invoked)
- Any shell command like `ls /home/dai` works because it's absolute path

#### Part 2: Filesystem Tools Also Use `Path.cwd()`

**File:** `relay/utils/paths.py`

```python
def resolve_path(file_path: Path) -> Path:
    """Relative paths are resolved against Path.cwd()."""
    return Path(file_path).resolve()
```

This means:
- Relative paths like `../` can escape to parent directories
- The configured `working_dir` is informational only, not enforced
- Tools can read/write anywhere the relay process has permissions

#### Part 3: Sandboxing Doesn't Help Unsandboxed Tools

The general agent configuration (`relay/resources/configs/default/agents/general.yml`) does **NOT** include `impl:terminal:*`:

```yaml
tools:
  patterns:
    - impl:web:*
    - impl:file_system:read_file
    - impl:file_system:glob_files
    - impl:file_system:grep_files
    - impl:file_system:ls
    - internal:*:*
    # NOTE: run_command is NOT listed
```

So `run_command` is not available to the general agent... **unless accessed via the catalog meta-tool** or **unless the LLM calls an MCP tool that then calls run_command**.

### Why This Design Choice?

The current behavior suggests intentional design decisions:

1. **Tool access is declarative and intent-based**: Patterns in YAML declare which tools an agent can use. By default, `run_command` is excluded (conservative).

2. **Working directory is advisory, not enforcing**: The `working_dir` parameter in `AgentContext` is passed to tools and prompt templates for context, but is **not enforced as a security boundary**. This allows tools to operate across the entire filesystem when needed.

3. **Sandboxing is opt-in per tool**: Individual tools can be configured to run sandboxed, but it requires:
   - Explicit agent config with `sandbox.enabled: true`
   - Explicit tool patterns in `sandbox.profiles`
   - A sandbox backend (bubblewrap on Linux only)

4. **MCP and Catalog tools bypass tool filtering**: The `fetch_tools`, `get_tool`, and `run_tool` meta-tools can invoke any tool in the catalog, regardless of the agent's configured patterns. This is a "secondary tool surface" for discovery.

---

## 4. Files Related to Sandboxing and Security

### Core Sandboxing
| File | Purpose |
|------|---------|
| `relay/sandboxes/backend.py` | Abstract SandboxBackend protocol |
| `relay/sandboxes/bubblewrap.py` | Linux bubblewrap implementation |
| `relay/sandboxes/worker.py` | Subprocess entry point inside sandbox |
| `relay/sandboxes/constants.py` | Sandboxing constants (syscalls, env vars, paths) |
| `relay/sandboxes/factory.py` | Factory for creating backends and bindings |

### Configuration
| File | Purpose |
|------|---------|
| `relay/configs/sandbox.py` | Pydantic models for sandbox config |
| `relay/configs/agent.py` | Agent config with embedded sandbox section |

### Middleware & Integration
| File | Purpose |
|------|---------|
| `relay/middlewares/sandbox.py` | SandboxMiddleware that intercepts tool calls |
| `relay/middlewares/approval.py` | Approval middleware (works with sandbox) |

### Tool Implementation
| File | Purpose |
|------|---------|
| `relay/tools/impl/terminal.py` | `run_command` tool (no working-dir enforcement) |
| `relay/tools/impl/filesystem/rw.py` | Read/write tools |
| `relay/tools/impl/filesystem/ls.py` | Directory listing |
| `relay/utils/paths.py` | Path resolution (uses `Path.cwd()`) |

### Agent & Runtime
| File | Purpose |
|------|---------|
| `relay/agents/factory.py` | AgentFactory (tool pattern filtering) |
| `relay/agents/context.py` | AgentContext (carries working_dir) |
| `relay/agents/react_agent.py` | Creates agent with middleware stack |
| `relay/tools/factory.py` | Tool registry with module maps |

### Test Files
| File | Purpose |
|------|---------|
| `tests/sandboxes/test_worker.py` | Worker module tests |
| `tests/sandboxes/test_factory.py` | Sandbox factory tests |
| `tests/sandboxes/test_bubblewrap.py` | BubblewrapBackend tests |
| `tests/middlewares/test_sandbox.py` | SandboxMiddleware tests |

---

## 5. Key Design Decisions & Comments

### 1. Module Allowlist Prevents Arbitrary Imports

**File:** `relay/sandboxes/constants.py`, line 13-14

```python
# Only tool modules under this prefix are importable inside the sandbox.
# This prevents the sandboxed process from importing arbitrary code.
ALLOWED_MODULE_PREFIX = "relay.tools."
```

**Implication:** A tool running inside sandbox cannot import arbitrary libraries or system modules. This is enforced in `worker.py` lines 60-64.

### 2. Seccomp Filtering is Optional

**File:** `relay/sandboxes/bubblewrap.py`, lines 34-43

```python
SECCOMP_AVAILABLE = False
try:
    import seccomp  # type: ignore[import-not-found]
    SECCOMP_AVAILABLE = True
except ImportError:
    pass
```

**Implication:** If `pyseccomp` is not installed, the sandbox still works but without BPF-based syscall filtering. Dangerous syscalls are not blocked.

### 3. Sandbox Command Is Wrapped in Bash for Fd-Passing

**File:** `relay/sandboxes/bubblewrap.py`, lines 160-167

```python
return [
    "bash",
    "-c",
    f'exec 3<{shlex.quote(str(filter_path))} && bwrap "$@"',
    "--",
    "--seccomp",
    "3",
] + base_cmd[1:]
```

**Implication:** bwrap expects seccomp BPF filter as an open file descriptor. Bash wrapper enables passing fd #3 to bwrap.

### 4. Working Directory Behavior Depends on Config

**File:** `relay/sandboxes/backend.py`, lines 191-195

```python
cwd = (
    str(self.working_dir)
    if self.includes_working_dir
    else str(Path(sys.executable).parent)
)
```

**Implication:** If sandbox config does NOT include `"."` in read/write paths, subprocess runs in the Python bin directory instead of project directory. This prevents filesystem access even to the project.

### 5. Filesystem Paths Are Glob-Expanded at Runtime

**File:** `relay/sandboxes/bubblewrap.py`, lines 51-71

```python
def _expand_pattern(pattern: str, working_dir: Path) -> list[Path]:
    """Expand a path pattern (possibly with globs or ~) to real paths."""
    expanded = Path(pattern).expanduser()
    base_path = expanded if expanded.is_absolute() else working_dir / expanded
    # ... handle glob patterns
```

**Implication:** Patterns like `~/.ssh` and `/tmp/*.log` are evaluated when sandbox is created, not stored as patterns inside the sandbox.

### 6. Session.md Already Documents the Working Directory Issue

**File:** `SESSION.md`, lines 7-11

```markdown
- `--working-dir` is not an enforced execution boundary: filesystem tools
  resolve paths via `Path.cwd()` in `relay/utils/paths.py`, and
  `relay/tools/impl/terminal.py` does not pass `cwd` to
  `create_subprocess_shell()`. Tool execution can therefore escape the selected
  project directory.
```

This is a **known limitation** documented by the project maintainers.

---

## 6. Attack Surface Analysis

### Sandboxed Tool Execution (Protected)

If `run_command` is configured to run sandboxed:
- ✅ Filesystem restricted to configured paths
- ✅ Network isolated (if `network.remote` is empty)
- ✅ Syscalls filtered (if seccomp available)
- ✅ Cannot import arbitrary Python modules
- ✅ Process namespaces isolated (PID, IPC, UTS, user)

### Unsandboxed Tool Execution (Not Protected)

If `run_command` runs unsandboxed (default):
- ❌ Can access any filesystem path
- ❌ Can execute any command
- ❌ Can make network connections
- ❌ Can import any Python module
- ❌ Inherits relay process permissions

### Catalog Meta-Tools (Potential Bypass)

The `run_tool` catalog meta-tool:
- Accepts `tool_name` and `tool_args` as parameters
- Looks up tool from `AgentContext.tool_catalog`
- Invokes tool directly (bypasses pattern filtering)
- **Note:** Catalog is not populated by default CLI, only used in special modes

---

## 7. Summary

### Sandboxing Infrastructure: ✅ Well-Designed

- OS-level isolation via bubblewrap
- JSON protocol minimizes attack surface
- Module allowlist prevents arbitrary imports
- Seccomp filtering blocks dangerous syscalls
- Configurable per-tool via patterns

### Boundary Enforcement: ❌ Not Implemented

- `run_command` tool does not accept or use `cwd` parameter
- Filesystem tools resolve paths via `Path.cwd()`, not agent `working_dir`
- `working_dir` is advisory/informational, not enforced
- **Session.md already documents this limitation (lines 7-11)**

### Design Intent

The current architecture appears intentionally designed:
1. **Declarative tool access** via patterns (conservative by default)
2. **Advisory working directory** for context/templates
3. **Opt-in sandboxing** per agent/tool
4. **Full filesystem access** for tools that need it (scripting, package management, etc.)

This is appropriate for an **AI coding assistant** where the agent needs broad filesystem and shell access by design.

---

## References

- **Main Sandbox Files:** `relay/sandboxes/backend.py`, `relay/sandboxes/bubblewrap.py`, `relay/sandboxes/worker.py`
- **Configuration:** `relay/configs/sandbox.py`, `relay/configs/agent.py`
- **Command Execution:** `relay/tools/impl/terminal.py`
- **Documentation:** `SESSION.md` (lines 7-11), `AGENTS.md`
- **Tests:** `tests/sandboxes/`, `tests/middlewares/test_sandbox.py`


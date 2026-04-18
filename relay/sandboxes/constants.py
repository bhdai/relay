"""Non-configurable sandbox constants.

These values are shared by backends and the worker module.  They are
kept separate from config classes to avoid circular imports.
"""

# ==============================================================================
# Worker
# ==============================================================================

WORKER_MODULE = "relay.sandboxes.worker"

# Only tool modules under this prefix are importable inside the sandbox.
# This prevents the sandboxed process from importing arbitrary code.
ALLOWED_MODULE_PREFIX = "relay.tools."

# ==============================================================================
# Output limits
# ==============================================================================

MAX_STDOUT = 10 * 1024 * 1024  # 10 MB
MAX_STDERR = 1 * 1024 * 1024  # 1 MB

# ==============================================================================
# Bubblewrap (Linux)
# ==============================================================================

BWRAP_AF_UNIX = 1
BWRAP_PTRACE_TRACEME = 0

# Syscalls to block via seccomp (beyond ptrace / AF_UNIX).
# These could allow sandbox escape or host system manipulation.
BWRAP_BLOCKED_SYSCALLS = (
    # Cross-process memory access
    "process_vm_readv",
    "process_vm_writev",
    # Kernel keyring (credential storage)
    "add_key",
    "request_key",
    "keyctl",
    # Kernel module loading
    "init_module",
    "finit_module",
    "delete_module",
    # System state manipulation
    "reboot",
    "kexec_load",
    "kexec_file_load",
    "swapon",
    "swapoff",
    # Can disable ASLR
    "personality",
    # eBPF (potential kernel exploit vector)
    "bpf",
    # Namespace manipulation (already isolated but defense in depth)
    "setns",
    "unshare",
)

# ==============================================================================
# Sandbox environment
# ==============================================================================

# Base environment for sandboxed processes.
# HOME is set per-backend to the working directory.
SANDBOX_ENV_BASE = {
    "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
    "LANG": "en_US.UTF-8",
    "TERM": "xterm-256color",
    "PYTHONUNBUFFERED": "1",
}

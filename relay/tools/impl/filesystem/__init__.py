"""Filesystem tools package.

Re-exports all public names so that ``from relay.tools.impl.filesystem import …``
works after the move to the impl sub-package.
"""

from relay.tools import walk_files
from relay.tools.impl.filesystem.glob import (
    _glob_match,
    glob_files,
)
from relay.tools.impl.filesystem.grep import (
    _grep_match,
    grep_files,
)
from relay.tools.impl.filesystem.ls import (
    _collect_files,
    _render_tree,
    ls,
)
from relay.tools.impl.filesystem.rw import (
    EditOperation,
    MoveOperation,
    _apply_edits,
    _find_match,
    _paginate_file,
    create_dir,
    delete_file,
    edit_file,
    move_file,
    read_file,
    write_file,
)

FILE_SYSTEM_TOOLS = [
    read_file,
    write_file,
    edit_file,
    create_dir,
    move_file,
    delete_file,
    glob_files,
    grep_files,
    ls,
]


# ==============================================================================
# Tool permission configuration
# ==============================================================================
#
# Read-only discovery tools (read_file, glob_files, grep_files, ls) use their
# respective permission keys so that fine-grained rules (e.g. "read: *.env →
# ask") work correctly.  The "always" pattern is "*" — once you allow reading
# you allow all reads, which matches Opencode's default behaviour.
#
# Write/mutating tools (write_file, edit_file, create_dir, move_file,
# delete_file) all use the "edit" permission key.  File-path-specific patterns
# let the ruleset distinguish individual files if needed.

for _tool in (read_file, glob_files, grep_files, ls):
    _perm_key = {
        "read_file": "read",
        "glob_files": "glob",
        "grep_files": "grep",
        "ls": "list",
    }[_tool.name]
    # Use "file_path" for read_file/ls, "pattern" for glob_files/grep_files.
    _arg_key = "file_path" if _tool.name in ("read_file", "ls") else "pattern"
    # ls uses dir_path, not file_path.
    if _tool.name == "ls":
        _arg_key = "dir_path"
    _captured_key = _arg_key  # capture for lambda closure
    _captured_perm = _perm_key
    _tool.metadata = {
        **((_tool.metadata or {})),
        "permission_config": {
            "permission": _captured_perm,
            "patterns_fn": (lambda args, k=_captured_key: [args.get(k, "*")]),
            # Once reading is approved, all reads are approved.
            "always_fn": lambda args: ["*"],
            "metadata_fn": (lambda args, k=_captured_key: {k: args.get(k, "")}),
        },
    }

for _tool in (write_file, edit_file, create_dir, move_file, delete_file):
    # Determine the primary path argument name per tool.
    _path_arg = {
        "write_file": "file_path",
        "edit_file": "file_path",
        "create_dir": "dir_path",
        "move_file": "source_path",
        "delete_file": "file_path",
    }[_tool.name]
    _captured_path_arg = _path_arg
    _tool.metadata = {
        **((_tool.metadata or {})),
        "permission_config": {
            "permission": "edit",
            "patterns_fn": (
                lambda args, k=_captured_path_arg: [args.get(k, "*")]
            ),
            # Once any edit is approved, all workspace edits are approved
            # (matching Opencode's default behaviour).
            "always_fn": lambda args: ["*"],
            "metadata_fn": (
                lambda args, k=_captured_path_arg: {"filepath": args.get(k, "")}
            ),
        },
    }

__all__ = [
    "EditOperation",
    "MoveOperation",
    "FILE_SYSTEM_TOOLS",
    "_apply_edits",
    "_collect_files",
    "_find_match",
    "_glob_match",
    "_grep_match",
    "_paginate_file",
    "_render_tree",
    "walk_files",
    "create_dir",
    "delete_file",
    "edit_file",
    "ls",
    "glob_files",
    "grep_files",
    "move_file",
    "read_file",
    "write_file",
]

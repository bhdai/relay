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


# Read-only discovery helpers should not require approval prompts.
for _tool in (read_file, glob_files, grep_files, ls):
    metadata = _tool.metadata or {}
    approval_config = metadata.get("approval_config", {})
    metadata["approval_config"] = {
        **approval_config,
        "always_approve": True,
    }
    _tool.metadata = metadata

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

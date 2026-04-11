"""Filesystem tools package.

Re-exports all public names so that ``from relay.tools.filesystem import …``
continues to work after the module-to-package conversion.
"""

from relay.tools import walk_files
from relay.tools.filesystem.glob import (
    _glob_match,
    glob_files,
)
from relay.tools.filesystem.grep import (
    _grep_match,
    grep_files,
)
from relay.tools.filesystem.ls import (
    _collect_files,
    _render_tree,
    ls,
)
from relay.tools.filesystem.rw import (
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

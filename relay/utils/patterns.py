"""Pattern matching utilities with negative pattern support.

Adapted from langrepl's ``utils/patterns.py``.  These are used by the
tool factory and agent factory to resolve declarative tool references
like ``impl:file_system:*`` or ``!impl:terminal:*``.
"""

from __future__ import annotations

from collections.abc import Callable
from fnmatch import fnmatch


def matches_patterns(
    patterns: list[str],
    matcher: Callable[[str], bool],
) -> bool:
    """Match against patterns with negative (``!pattern``) support.

    Returns ``True`` if the item matches at least one positive pattern
    AND matches zero negative patterns.  Returns ``False`` if there
    are no positive patterns at all.
    """
    positives = [p for p in patterns if not p.startswith("!")]
    negatives = [p[1:] for p in patterns if p.startswith("!")]

    if not positives:
        return False

    return any(matcher(p) for p in positives) and not any(
        matcher(p) for p in negatives
    )


def two_part_matcher(
    name: str,
    module: str,
    on_invalid: Callable[[str], None] | None = None,
) -> Callable[[str], bool]:
    """Return a matcher for 2-part patterns (``module:name``).

    Used to filter impl and internal tools where the reference format
    is ``module:tool_name``, e.g. ``file_system:read_file`` or
    ``terminal:*``.
    """

    def match(pattern: str) -> bool:
        parts = pattern.split(":")
        if len(parts) != 2:
            if on_invalid:
                on_invalid(pattern)
            return False
        mod_pattern, name_pattern = parts
        return fnmatch(module, mod_pattern) and fnmatch(name, name_pattern)

    return match

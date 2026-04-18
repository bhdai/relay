"""Permission evaluator — pure functions, no side effects.

``evaluate`` is the core of the permission engine.  It merges zero or
more ``Ruleset`` objects into a flat ordered list and applies
last-match-wins semantics to find the most specific applicable rule.

``wildcard_match`` is a lightweight glob-style matcher ported from
Opencode's ``Wildcard.match`` (TypeScript).  It supports ``*``
(any sequence), ``?`` (single character), and the special ``' *'``
optional trailing suffix.
"""

from __future__ import annotations

import re

from relay.permission.schema import PermissionRule, Ruleset

# ==============================================================================
# Wildcard matcher
# ==============================================================================

# Characters that have special meaning in Python ``re`` and must be
# escaped before we substitute our own wildcard tokens.
_REGEX_SPECIALS = re.compile(r"([.+^${}()|\[\]\\])")


def wildcard_match(text: str, pattern: str) -> bool:
    """Return ``True`` if *text* matches *pattern*.

    Pattern syntax:

    - ``*``  — matches any sequence of characters (including ``/``).
    - ``?``  — matches exactly one character.
    - ``' *'`` (space followed by star) at the end of the pattern is
      treated as an *optional* suffix: the space and whatever follows
      it may be absent.  This means ``"git push *"`` matches both
      ``"git push"`` and ``"git push --force origin main"``.

    Matching is case-sensitive on Linux (mirrors the host filesystem).
    Path separators are normalised to ``/`` so that patterns written
    with forward slashes work on all platforms.

    Args:
        text: The concrete value to test, e.g. a command string or file path.
        pattern: A wildcard pattern, e.g. ``"git push *"``, ``"src/**"``, ``"*"``.
    """
    # Normalise path separators so that patterns always use ``/``.
    text = text.replace("\\", "/")
    pattern = pattern.replace("\\", "/")

    # Detect the trailing ``' *'`` optional-suffix convention.
    # ``"git push *"`` should match both ``"git push"`` (no args)
    # and ``"git push --force"`` (with args).
    if pattern.endswith(" *"):
        # Build two sub-patterns: one without the optional suffix and
        # one with the full wildcard expansion, then OR them together.
        base = pattern[:-2]  # strip " *"
        regex = _to_regex(base) + r"(?:\s.*)?" + "$"
    else:
        regex = _to_regex(pattern) + "$"

    return bool(re.match(regex, text))


def _to_regex(pattern: str) -> str:
    """Convert a plain wildcard pattern (no trailing-star handling) to regex.

    Escapes all regex metacharacters in *pattern*, then substitutes
    the wildcard tokens:

    - ``?`` (placeholder ``\x00``) → ``.``
    - ``*`` (placeholder ``\x01``) → ``.*``

    Args:
        pattern: A wildcard pattern string with ``*`` / ``?`` tokens.

    Returns:
        A ``^``-anchored regex string without a trailing ``$``.
    """
    # Pre-replace wildcard tokens with safe placeholders before escaping,
    # so that the ``re.escape`` call does not mangle them.
    pattern = pattern.replace("?", "\x00").replace("*", "\x01")

    # Escape all remaining regex metacharacters.
    pattern = _REGEX_SPECIALS.sub(r"\\\1", pattern)

    # Restore wildcards as their regex equivalents.
    pattern = pattern.replace("\x00", ".").replace("\x01", ".*")

    return "^" + pattern


# ==============================================================================
# Evaluate
# ==============================================================================

# Default rule returned when no rule in the merged ruleset matches.
# The explicit ``permission`` and ``pattern`` values are intentionally
# broad to signal "no rule was found, fall back to asking the user".
_DEFAULT_ASK = PermissionRule(permission="*", pattern="*", action="ask")


def evaluate(
    permission: str,
    pattern: str,
    *rulesets: Ruleset,
) -> PermissionRule:
    """Evaluate *permission* / *pattern* against merged *rulesets*.

    Rules are evaluated in order; the **last matching rule wins**.
    This mirrors Opencode's ``evaluate.ts`` and gives later (more
    specific) rules the ability to override earlier (more general) ones.

    Args:
        permission: The permission key to check, e.g. ``"bash"``, ``"edit"``.
        pattern: The concrete value for this request, e.g. a shell command
            or file path.
        *rulesets: Zero or more ``Ruleset`` objects (lists of
            ``PermissionRule``), merged in order.  Rules in later rulesets
            override rules in earlier ones because evaluation is
            last-match-wins.

    Returns:
        The last rule in the merged list whose ``permission`` and
        ``pattern`` fields both match — or ``_DEFAULT_ASK`` if no rule
        matched at all.
    """
    last_match: PermissionRule | None = None

    for ruleset in rulesets:
        for rule in ruleset:
            perm_matches = wildcard_match(permission, rule.permission)
            pat_matches = wildcard_match(pattern, rule.pattern)
            if perm_matches and pat_matches:
                last_match = rule

    return last_match if last_match is not None else _DEFAULT_ASK

"""Utility functions for loading YAML configs and prompt content."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


async def load_yaml_dir(dir_path: Path) -> list[dict]:
    """Load all ``.yml`` files from *dir_path* as a flat list of dicts.

    Each file may contain either a single mapping or a list of mappings.
    Files are processed in sorted order for determinism.
    """
    if not dir_path.exists():
        return []

    items: list[dict] = []
    yml_files = await asyncio.to_thread(lambda: sorted(dir_path.glob("*.yml")))

    for yml_file in yml_files:
        content = await asyncio.to_thread(yml_file.read_text)
        data = yaml.safe_load(content)

        if isinstance(data, list):
            items.extend(data)
        elif isinstance(data, dict):
            items.append(data)

    return items


async def load_yaml_file(file_path: Path, key: str) -> list[dict]:
    """Load items from a YAML file under the given *key*.

    Expects the file to contain a mapping with *key* pointing to a list
    of mappings, e.g.::

        agents:
          - name: general
            ...
    """
    if not file_path.exists():
        return []

    content = await asyncio.to_thread(file_path.read_text)
    data = yaml.safe_load(content)

    if not isinstance(data, dict):
        return []

    items = data.get(key, [])
    return items if isinstance(items, list) else []


async def load_prompt_content(
    base_path: Path,
    prompt: str | list[str] | None,
) -> str:
    """Load and concatenate prompt content from file path(s).

    Parameters
    ----------
    base_path:
        Directory that prompt file paths are resolved relative to.
    prompt:
        A single file path, a list of file paths, or literal prompt
        text.  Paths that resolve to existing files are read; anything
        else is kept as literal text.

    Returns
    -------
    str
        Concatenated prompt content separated by double newlines.
    """
    if not prompt:
        return ""

    if isinstance(prompt, str):
        prompt_path = base_path / prompt
        if prompt_path.exists() and prompt_path.is_file():
            return await asyncio.to_thread(prompt_path.read_text)
        # Treat as literal prompt text.
        return prompt

    if isinstance(prompt, list):
        parts: list[str] = []
        for entry in prompt:
            prompt_path = base_path / entry
            if prompt_path.exists() and prompt_path.is_file():
                content = await asyncio.to_thread(prompt_path.read_text)
                parts.append(content)
            else:
                parts.append(entry)
        return "\n\n".join(parts)

    return str(prompt)

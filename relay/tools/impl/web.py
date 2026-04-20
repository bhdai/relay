"""Web content extraction tool.

Uses ``trafilatura`` to download a webpage and extract its main content as
Markdown.
"""

from urllib.parse import urlparse

import trafilatura
from langchain_core.tools import ToolException, tool


@tool
async def fetch_web_content(url: str) -> str:
    """Fetch a webpage and return its main content as Markdown.

    Args:
        url: The URL to fetch.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
    except Exception as exc:
        raise ToolException(f"fetch web content from {url}") from exc

    if downloaded is None:
        raise ToolException(f"Could not download content from {url}")

    content = trafilatura.extract(downloaded, output_format="markdown")
    if not content:
        return f"No main content could be extracted from {url}"

    domain = urlparse(url).netloc
    return f"# Content from {domain}\n\n{content}"


WEB_TOOLS = [fetch_web_content]

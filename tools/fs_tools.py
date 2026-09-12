"""fs_tools.py

Core file-system tools for the LLM-Powered File System Assistant.

Reused unchanged (aside from ``WORKSPACE_ROOT``) from the original
"LLM-Powered File System Assistant" (Milestone 1) project. Every function is a
plain Python function that can be imported and called directly:

    from tools.fs_tools import read_file, list_files, write_file, search_in_file

    result = read_file("data/resumes/01_aria_carver.pdf")
    print(result["content"])

Design notes
------------
* Supported document types: PDF (.pdf), Word (.docx) and plain text (.txt/.md).
* Every function returns a structured, JSON-serializable dictionary (built from
  a pydantic model) so the results are easy to hand back to an LLM.
* Errors are handled gracefully. A single bad file never raises out of these
  functions -- instead a dictionary with ``success == False`` and a helpful
  ``error`` message is returned.
* All file access is constrained to the project workspace for safety.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# Third-party document parsers. Imported lazily-friendly at module load; if a
# parser is missing the relevant branch raises a clear DocumentParseError.
try:  # PDF support
    from pypdf import PdfReader
except Exception:  # pragma: no cover - handled at call time
    PdfReader = None  # type: ignore[assignment]

try:  # DOCX support
    from docx import Document
except Exception:  # pragma: no cover - handled at call time
    Document = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Configuration / constants
# ---------------------------------------------------------------------------

#: The workspace root. All tool file access is restricted to inside this path.
#: This file lives in ``<project_root>/tools/fs_tools.py``, so the project
#: root is one directory up.
WORKSPACE_ROOT: Path = Path(__file__).resolve().parent.parent

#: File extensions we know how to extract text from.
SUPPORTED_TEXT_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

#: Default number of characters of context shown around each search match.
DEFAULT_CONTEXT_CHARS = 80


# ---------------------------------------------------------------------------
# Internal errors (used only inside this module)
# ---------------------------------------------------------------------------


class ToolError(Exception):
    """Base class for controlled, message-carrying tool errors."""


class UnsupportedFileTypeError(ToolError):
    """Raised when a file extension is not one we can extract text from."""


class DocumentParseError(ToolError):
    """Raised when a supported document cannot be parsed."""


class OutsideWorkspaceError(ToolError):
    """Raised when a path resolves outside the allowed workspace."""


# ---------------------------------------------------------------------------
# Pydantic argument (input) models
# ---------------------------------------------------------------------------


class ReadFileArgs(BaseModel):
    """Validated arguments for :func:`read_file`."""

    filepath: str = Field(..., description="Path to the file to read.")


class ListFilesArgs(BaseModel):
    """Validated arguments for :func:`list_files`."""

    directory: str = Field(..., description="Directory to list files from.")
    extension: Optional[str] = Field(
        default=None,
        description="Optional extension filter, e.g. '.pdf' or 'pdf'.",
    )


class WriteFileArgs(BaseModel):
    """Validated arguments for :func:`write_file`."""

    filepath: str = Field(..., description="Path of the file to write.")
    content: str = Field(..., description="Text content to write into the file.")


class SearchInFileArgs(BaseModel):
    """Validated arguments for :func:`search_in_file`."""

    filepath: str = Field(..., description="Path of the file to search in.")
    keyword: str = Field(..., description="Keyword to search for (case-insensitive).")
    context_chars: int = Field(
        default=DEFAULT_CONTEXT_CHARS,
        ge=0,
        le=1000,
        description="Characters of surrounding context to include per match.",
    )


# ---------------------------------------------------------------------------
# Pydantic result (output) models
# ---------------------------------------------------------------------------


class ReadFileResult(BaseModel):
    """Structured result of :func:`read_file`."""

    success: bool
    filepath: str
    filename: Optional[str] = None
    file_type: Optional[str] = None
    content: Optional[str] = None
    size_bytes: Optional[int] = None
    modified: Optional[str] = None
    error: Optional[str] = None


class FileInfo(BaseModel):
    """Metadata for a single file returned by :func:`list_files`."""

    name: str
    path: str
    size_bytes: int
    modified: str
    extension: str


class WriteFileResult(BaseModel):
    """Structured result of :func:`write_file`."""

    success: bool
    filepath: str
    message: Optional[str] = None
    error: Optional[str] = None


class SearchMatch(BaseModel):
    """A single keyword match with surrounding context."""

    position: int = Field(..., description="Character offset of the match.")
    context: str = Field(..., description="Surrounding text around the match.")


class SearchInFileResult(BaseModel):
    """Structured result of :func:`search_in_file`."""

    success: bool
    filepath: str
    keyword: str
    total_matches: int = 0
    matches: List[SearchMatch] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_within_workspace(path_str: str) -> Path:
    """Resolve ``path_str`` and ensure it stays inside :data:`WORKSPACE_ROOT`.

    Relative paths are interpreted relative to the workspace root. Any attempt
    to escape the workspace (e.g. ``../../etc/passwd``) raises
    :class:`OutsideWorkspaceError`.
    """

    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = WORKSPACE_ROOT / candidate
    resolved = candidate.resolve()

    try:
        resolved.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise OutsideWorkspaceError(
            f"Access denied: '{path_str}' is outside the allowed workspace."
        ) from exc
    return resolved


def _format_mtime(path: Path) -> str:
    """Return the file's last-modified time as an ISO-8601 string."""

    ts = path.stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat()


def _extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF file using :mod:`pypdf`."""

    if PdfReader is None:
        raise DocumentParseError(
            "PDF support requires the 'pypdf' package. Install it via requirements.txt."
        )
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:  # noqa: BLE001 - convert to controlled error
        raise DocumentParseError(f"Failed to parse PDF: {exc}") from exc
    return "\n".join(pages).strip()


def _extract_docx_text(path: Path) -> str:
    """Extract text from a DOCX file using :mod:`python-docx`."""

    if Document is None:
        raise DocumentParseError(
            "DOCX support requires the 'python-docx' package. "
            "Install it via requirements.txt."
        )
    try:
        document = Document(str(path))
        paragraphs = [para.text for para in document.paragraphs]
    except Exception as exc:  # noqa: BLE001 - convert to controlled error
        raise DocumentParseError(f"Failed to parse DOCX: {exc}") from exc
    return "\n".join(paragraphs).strip()


def _extract_txt_text(path: Path) -> str:
    """Read a plain-text/markdown file as UTF-8."""

    try:
        return path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        # Fall back to a lenient decode so odd encodings don't crash us.
        return path.read_text(encoding="utf-8", errors="replace").strip()


def _extract_text(path: Path) -> str:
    """Extract text from a supported document based on its extension.

    Raises
    ------
    UnsupportedFileTypeError
        If the extension is not supported.
    DocumentParseError
        If a supported document cannot be parsed.
    """

    ext = path.suffix.lower()
    if ext == ".pdf":
        return _extract_pdf_text(path)
    if ext == ".docx":
        return _extract_docx_text(path)
    if ext in (".txt", ".md"):
        return _extract_txt_text(path)
    raise UnsupportedFileTypeError(
        f"Unsupported file type '{ext}'. Supported types: "
        f"{', '.join(sorted(SUPPORTED_TEXT_EXTENSIONS))}."
    )


def _normalize_extension(extension: Optional[str]) -> Optional[str]:
    """Normalize an extension filter so 'pdf' and '.pdf' behave the same."""

    if extension is None:
        return None
    ext = extension.strip().lower()
    if not ext:
        return None
    if not ext.startswith("."):
        ext = "." + ext
    return ext


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def read_file(filepath: str) -> dict:
    """Read a resume/document file and extract its text content.

    Supports PDF, DOCX and TXT/MD files.

    Parameters
    ----------
    filepath:
        Path to the file (relative to the workspace or absolute inside it).

    Returns
    -------
    dict
        A dictionary matching :class:`ReadFileResult` with ``success``,
        ``filepath``, ``filename``, ``file_type``, ``content``, ``size_bytes``,
        ``modified`` and, on failure, an ``error`` message.
    """

    try:
        path = _resolve_within_workspace(filepath)
    except OutsideWorkspaceError as exc:
        return ReadFileResult(success=False, filepath=filepath, error=str(exc)).model_dump()

    if not path.exists():
        return ReadFileResult(
            success=False, filepath=filepath, error=f"File not found: {filepath}"
        ).model_dump()

    if not path.is_file():
        return ReadFileResult(
            success=False, filepath=filepath, error=f"Not a file: {filepath}"
        ).model_dump()

    try:
        content = _extract_text(path)
        stat = path.stat()
        return ReadFileResult(
            success=True,
            filepath=str(path),
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            content=content,
            size_bytes=stat.st_size,
            modified=_format_mtime(path),
        ).model_dump()
    except (UnsupportedFileTypeError, DocumentParseError) as exc:
        return ReadFileResult(success=False, filepath=filepath, error=str(exc)).model_dump()
    except PermissionError:
        return ReadFileResult(
            success=False,
            filepath=filepath,
            error=f"Permission denied while reading: {filepath}",
        ).model_dump()
    except OSError as exc:
        return ReadFileResult(
            success=False, filepath=filepath, error=f"OS error while reading file: {exc}"
        ).model_dump()


def list_files(directory: str, extension: Optional[str] = None) -> list:
    """List files in a directory, optionally filtered by extension.

    Parameters
    ----------
    directory:
        Directory to list (relative to the workspace or absolute inside it).
    extension:
        Optional extension filter. ``".pdf"`` and ``"pdf"`` are both accepted.

    Returns
    -------
    list
        A list of file-metadata dictionaries (see :class:`FileInfo`). If the
        directory does not exist or cannot be read, a single-element list
        containing an error dictionary is returned instead, so the caller is
        always informed without an exception being raised.
    """

    try:
        dir_path = _resolve_within_workspace(directory)
    except OutsideWorkspaceError as exc:
        return [{"success": False, "error": str(exc)}]

    if not dir_path.exists():
        return [{"success": False, "error": f"Directory not found: {directory}"}]

    if not dir_path.is_dir():
        return [{"success": False, "error": f"Not a directory: {directory}"}]

    wanted_ext = _normalize_extension(extension)

    results: List[dict] = []
    try:
        for entry in sorted(dir_path.iterdir()):
            if not entry.is_file():
                continue  # Do not include directories.
            if wanted_ext is not None and entry.suffix.lower() != wanted_ext:
                continue
            stat = entry.stat()
            results.append(
                FileInfo(
                    name=entry.name,
                    path=str(entry),
                    size_bytes=stat.st_size,
                    modified=_format_mtime(entry),
                    extension=entry.suffix.lower().lstrip("."),
                ).model_dump()
            )
    except PermissionError:
        return [{"success": False, "error": f"Permission denied for directory: {directory}"}]
    except OSError as exc:
        return [{"success": False, "error": f"OS error while listing directory: {exc}"}]

    return results


def write_file(filepath: str, content: str) -> dict:
    """Write text ``content`` to ``filepath`` (UTF-8), creating parent dirs.

    This tool is primarily intended for creating summary ``.txt`` or ``.md``
    files.

    Parameters
    ----------
    filepath:
        Destination path (relative to the workspace or absolute inside it).
    content:
        Text to write.

    Returns
    -------
    dict
        A dictionary matching :class:`WriteFileResult` with ``success``,
        ``filepath``, ``message`` and, on failure, an ``error`` message.
    """

    try:
        path = _resolve_within_workspace(filepath)
    except OutsideWorkspaceError as exc:
        return WriteFileResult(success=False, filepath=filepath, error=str(exc)).model_dump()

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return WriteFileResult(
            success=True,
            filepath=str(path),
            message=f"Wrote {len(content)} characters to {path.name}.",
        ).model_dump()
    except PermissionError:
        return WriteFileResult(
            success=False,
            filepath=filepath,
            error=f"Permission denied while writing: {filepath}",
        ).model_dump()
    except OSError as exc:
        return WriteFileResult(
            success=False, filepath=filepath, error=f"OS error while writing file: {exc}"
        ).model_dump()


def search_in_file(
    filepath: str, keyword: str, context_chars: int = DEFAULT_CONTEXT_CHARS
) -> dict:
    """Search a document for ``keyword`` (case-insensitive) with context.

    Supports PDF, DOCX and TXT/MD files using the same extraction logic as
    :func:`read_file`.

    Parameters
    ----------
    filepath:
        Path of the file to search.
    keyword:
        Keyword to search for. Matching is case-insensitive.
    context_chars:
        Number of characters of surrounding text to include on each side of a
        match.

    Returns
    -------
    dict
        A dictionary matching :class:`SearchInFileResult` with ``success``,
        ``filepath``, ``keyword``, ``total_matches``, ``matches`` and, on
        failure, an ``error`` message.
    """

    if not keyword:
        return SearchInFileResult(
            success=False,
            filepath=filepath,
            keyword=keyword,
            error="Keyword must not be empty.",
        ).model_dump()

    try:
        path = _resolve_within_workspace(filepath)
    except OutsideWorkspaceError as exc:
        return SearchInFileResult(
            success=False, filepath=filepath, keyword=keyword, error=str(exc)
        ).model_dump()

    if not path.exists() or not path.is_file():
        return SearchInFileResult(
            success=False,
            filepath=filepath,
            keyword=keyword,
            error=f"File not found: {filepath}",
        ).model_dump()

    try:
        text = _extract_text(path)
    except (UnsupportedFileTypeError, DocumentParseError) as exc:
        return SearchInFileResult(
            success=False, filepath=filepath, keyword=keyword, error=str(exc)
        ).model_dump()
    except PermissionError:
        return SearchInFileResult(
            success=False,
            filepath=filepath,
            keyword=keyword,
            error=f"Permission denied while reading: {filepath}",
        ).model_dump()
    except OSError as exc:
        return SearchInFileResult(
            success=False,
            filepath=filepath,
            keyword=keyword,
            error=f"OS error while reading file: {exc}",
        ).model_dump()

    haystack = text.lower()
    needle = keyword.lower()

    matches: List[SearchMatch] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            break
        ctx_start = max(0, idx - context_chars)
        ctx_end = min(len(text), idx + len(keyword) + context_chars)
        snippet = text[ctx_start:ctx_end].replace("\n", " ").strip()
        if ctx_start > 0:
            snippet = "..." + snippet
        if ctx_end < len(text):
            snippet = snippet + "..."
        matches.append(SearchMatch(position=idx, context=snippet))
        start = idx + len(needle)

    return SearchInFileResult(
        success=True,
        filepath=str(path),
        keyword=keyword,
        total_matches=len(matches),
        matches=matches,
    ).model_dump()


# ---------------------------------------------------------------------------
# Manual smoke test: `python tools/fs_tools.py`
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("Listing data/resumes/:")
    print(json.dumps(list_files("data/resumes"), indent=2))

    print("\nListing only .pdf resumes:")
    print(json.dumps(list_files("data/resumes", "pdf"), indent=2))

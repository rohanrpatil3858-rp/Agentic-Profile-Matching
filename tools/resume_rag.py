"""
resume_rag.py
=============

Core Retrieval-Augmented-Generation (RAG) building block, reused from the
"RAG based profile matching v1" project (Milestone 2). Logic is unchanged;
only the default resume/Chroma directories are adapted to this project's
``data/`` layout.

Responsibilities
----------------
1. Recursively load resume PDF files from ``data/resumes/``.
2. Extract raw text using :mod:`pypdf`.
3. Extract lightweight structured metadata from each resume
   (candidate name, skills, experience years, education, resume path).
4. Perform section-aware chunking that keeps logical resume sections
   (Summary, Skills, Experience, Education, Projects) intact where possible.
5. Embed chunks with ``sentence-transformers/all-MiniLM-L6-v2``.
6. Persist chunks + metadata + embeddings in a persistent ChromaDB collection.

The module is deliberately dependency-light and defensive: corrupted PDFs are
skipped with a warning instead of crashing the whole ingestion run.
"""

from __future__ import annotations

import os
import re
import glob
import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

try:
    from pypdf import PdfReader
except ImportError as exc:  # pragma: no cover - clear error for missing dep
    raise ImportError(
        "pypdf is required. Install project requirements with "
        "`pip install -r requirements.txt`."
    ) from exc


# --------------------------------------------------------------------------- #
# Constants / configuration
# --------------------------------------------------------------------------- #

# Folder that stores resume PDFs (searched recursively).
RESUME_DIR = os.path.join("data", "resumes")

# Persistent ChromaDB location and collection name.
CHROMA_DIR = os.path.join("data", "chroma_db")
COLLECTION_NAME = "resumes"

# Embedding model shared by ingestion and querying.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Canonical resume section headers we try to preserve during chunking.
# Each canonical name maps to a set of case-insensitive header aliases.
SECTION_ALIASES: Dict[str, List[str]] = {
    "Summary": ["summary", "professional summary", "profile", "objective", "about"],
    "Skills": ["skills", "technical skills", "core competencies", "technologies"],
    "Experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
    ],
    "Education": ["education", "academic background", "qualifications"],
    "Projects": ["projects", "personal projects", "key projects", "selected projects"],
    "Certifications": ["certifications", "certificates", "licenses"],
}

# A curated skill vocabulary used for keyword extraction. Extend as needed.
SKILL_VOCABULARY: List[str] = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "scala", "kotlin", "swift", "r", "matlab", "sql",
    "nosql", "postgresql", "mysql", "mongodb", "redis", "cassandra",
    "html", "css", "react", "angular", "vue", "node.js", "nodejs", "express",
    "django", "flask", "fastapi", "spring", "spring boot", ".net", "rails",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data analysis", "statistics",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "matplotlib", "opencv", "hugging face", "huggingface",
    "transformers", "llm", "rag", "langchain", "openai",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "ci/cd", "git", "linux",
    "spark", "hadoop", "kafka", "airflow", "etl", "databricks", "snowflake",
    "tableau", "power bi", "excel", "rest", "graphql", "grpc",
    "microservices", "agile", "scrum", "devops", "mlops",
]


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class ResumeMetadata:
    """Structured metadata extracted from a single resume."""

    candidate_name: str
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: str = ""
    resume_path: str = ""

    def to_chroma_metadata(self) -> Dict[str, str]:
        """Flatten metadata into ChromaDB-compatible scalar values.

        ChromaDB only accepts ``str``, ``int``, ``float`` and ``bool`` metadata
        values, so lists are serialised into comma-separated strings.
        """
        return {
            "candidate_name": self.candidate_name,
            "skills": ", ".join(self.skills),
            "experience_years": float(self.experience_years),
            "education": self.education,
            "resume_path": self.resume_path,
        }


@dataclass
class ResumeChunk:
    """A single embeddable text chunk plus its provenance metadata."""

    chunk_id: str
    text: str
    section: str
    metadata: ResumeMetadata


# --------------------------------------------------------------------------- #
# PDF loading & text extraction
# --------------------------------------------------------------------------- #

def find_resume_files(resume_dir: str = RESUME_DIR) -> List[str]:
    """Recursively collect all PDF file paths under ``resume_dir``.

    Parameters
    ----------
    resume_dir:
        Root directory to search.

    Returns
    -------
    list[str]
        Sorted list of relative PDF paths.
    """
    pattern = os.path.join(resume_dir, "**", "*.pdf")
    return sorted(glob.glob(pattern, recursive=True))


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using :mod:`pypdf`.

    Corrupted or unreadable pages are skipped so that a single bad page does
    not discard the entire document.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    str
        Concatenated text from all readable pages. Empty string on failure.
    """
    try:
        reader = PdfReader(pdf_path)
    except Exception as exc:  # noqa: BLE001 - defensive: any pypdf error
        print(f"[WARN] Could not open '{pdf_path}': {exc}")
        return ""

    pages_text: List[str] = []
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        except Exception as exc:  # noqa: BLE001 - skip bad page, keep going
            print(f"[WARN] Failed to read page {page_num} of '{pdf_path}': {exc}")
            continue

    return "\n".join(pages_text).strip()


# --------------------------------------------------------------------------- #
# Metadata extraction
# --------------------------------------------------------------------------- #

def _guess_candidate_name(text: str, pdf_path: str) -> str:
    """Best-effort guess of the candidate name.

    Strategy:
    1. Use the first non-empty line if it looks like a person's name
       (1-4 capitalised-ish words, no digits/@).
    2. Fall back to a cleaned version of the file name.
    """
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        # Reject obvious non-name lines.
        if any(ch.isdigit() for ch in candidate) or "@" in candidate:
            break
        words = candidate.split()
        if 1 <= len(words) <= 4 and len(candidate) <= 50:
            # Require it to look name-like (mostly alphabetic tokens).
            if all(re.match(r"^[A-Za-z.,'-]+$", w) for w in words):
                return candidate.title()
        break

    # Fallback: derive from filename, e.g. "john_doe.pdf" -> "John Doe".
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    stem = re.sub(r"[_\-]+", " ", stem)
    return stem.strip().title() or "Unknown Candidate"


def _extract_skills(text: str) -> List[str]:
    """Extract known skills from resume text using the curated vocabulary."""
    lowered = text.lower()
    found: List[str] = []
    for skill in SKILL_VOCABULARY:
        # Word-boundary match; escape regex meta characters (e.g. "c++").
        pattern = r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])"
        if re.search(pattern, lowered):
            found.append(skill)
    # De-duplicate while preserving order; normalise display casing.
    seen = set()
    unique: List[str] = []
    for skill in found:
        if skill not in seen:
            seen.add(skill)
            unique.append(skill)
    return unique


def _extract_experience_years(text: str) -> float:
    """Estimate total years of experience.

    Two heuristics are combined and the larger value is returned:
    1. Explicit phrases like "5+ years of experience".
    2. Span between the earliest and latest 4-digit years mentioned.
    """
    lowered = text.lower()

    explicit_years = 0.0
    for match in re.finditer(r"(\d{1,2})\s*\+?\s*years?", lowered):
        try:
            explicit_years = max(explicit_years, float(match.group(1)))
        except ValueError:
            continue

    span_years = 0.0
    years = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", text)]
    current_year = date.today().year
    years = [y for y in years if 1950 <= y <= current_year]
    if len(years) >= 2:
        span_years = float(max(years) - min(years))

    return max(explicit_years, span_years)


def _extract_education(text: str) -> str:
    """Extract a short education descriptor.

    Looks for common degree keywords near the education section and returns the
    first matching line; otherwise returns an empty string.
    """
    degree_keywords = [
        "ph.d", "phd", "doctorate", "master", "m.sc", "msc", "mba", "m.tech",
        "bachelor", "b.sc", "bsc", "b.tech", "b.e", "b.a", "associate",
    ]
    for line in text.splitlines():
        lowered = line.lower()
        if any(kw in lowered for kw in degree_keywords):
            cleaned = line.strip()
            if cleaned:
                return cleaned[:200]
    return ""


def extract_metadata(text: str, pdf_path: str) -> ResumeMetadata:
    """Build a :class:`ResumeMetadata` object from resume text."""
    return ResumeMetadata(
        candidate_name=_guess_candidate_name(text, pdf_path),
        skills=_extract_skills(text),
        experience_years=_extract_experience_years(text),
        education=_extract_education(text),
        resume_path=pdf_path.replace("\\", "/"),
    )


# --------------------------------------------------------------------------- #
# Section-aware chunking
# --------------------------------------------------------------------------- #

def _build_header_pattern() -> re.Pattern:
    """Compile a regex that matches any known section header alias."""
    aliases = []
    for alias_list in SECTION_ALIASES.values():
        aliases.extend(alias_list)
    # Longer aliases first so multi-word headers win over single words.
    aliases = sorted(set(aliases), key=len, reverse=True)
    alias_group = "|".join(re.escape(a) for a in aliases)
    # A header line: optional leading spaces, the alias, optional trailing colon.
    return re.compile(
        rf"^\s*(?P<header>{alias_group})\s*:?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )


_HEADER_PATTERN = _build_header_pattern()


def _canonical_section(header_text: str) -> str:
    """Map a matched header alias back to its canonical section name."""
    lowered = header_text.strip().lower()
    for canonical, aliases in SECTION_ALIASES.items():
        if lowered in aliases:
            return canonical
    return "Other"


def split_into_sections(text: str) -> List[Dict[str, str]]:
    """Split resume text into labelled sections.

    Returns a list of ``{"section": name, "text": body}`` dictionaries in the
    order they appear. Text before the first recognised header is labelled
    ``Header`` (usually contact info / name).
    """
    matches = list(_HEADER_PATTERN.finditer(text))
    if not matches:
        return [{"section": "General", "text": text.strip()}]

    sections: List[Dict[str, str]] = []

    # Preamble before the first header.
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append({"section": "Header", "text": preamble})

    for i, match in enumerate(matches):
        section_name = _canonical_section(match.group("header"))
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append({"section": section_name, "text": body})

    return sections


def _split_long_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split ``text`` into overlapping pieces no longer than ``max_chars``.

    Splitting prefers paragraph and sentence boundaries so chunks stay
    semantically coherent.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    # First try paragraph-based accumulation.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > max_chars:
            # Flush current buffer, then hard-split the oversized paragraph.
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.extend(_hard_split(para, max_chars, overlap))
            continue

        if len(current) + len(para) + 1 <= max_chars:
            current = f"{current}\n{para}".strip()
        else:
            if current:
                chunks.append(current.strip())
            current = para

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c]


def _hard_split(text: str, max_chars: int, overlap: int) -> List[str]:
    """Hard character window split with overlap for very long blocks."""
    chunks: List[str] = []
    start = 0
    step = max(1, max_chars - overlap)
    while start < len(text):
        chunks.append(text[start : start + max_chars].strip())
        start += step
    return [c for c in chunks if c]


def chunk_resume(
    text: str,
    metadata: ResumeMetadata,
    max_chars: int = 1200,
    overlap: int = 150,
) -> List[ResumeChunk]:
    """Produce section-aware chunks for a single resume.

    Each recognised section becomes one or more chunks. Oversized sections are
    split further with overlap while smaller sections stay intact.

    Parameters
    ----------
    text:
        Full resume text.
    metadata:
        Extracted metadata (attached to every chunk).
    max_chars:
        Soft maximum characters per chunk.
    overlap:
        Overlap (characters) applied when a section must be hard-split.
    """
    sections = split_into_sections(text)
    chunks: List[ResumeChunk] = []

    for section in sections:
        section_name = section["section"]
        pieces = _split_long_text(section["text"], max_chars, overlap)
        for idx, piece in enumerate(pieces):
            # Deterministic, collision-resistant unique ID per chunk.
            raw_id = f"{metadata.resume_path}::{section_name}::{idx}::{piece[:64]}"
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()
            # Prefix the section name to give the embedding useful context.
            chunk_text = f"[{section_name}]\n{piece}"
            chunks.append(
                ResumeChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    section=section_name,
                    metadata=metadata,
                )
            )

    return chunks


# --------------------------------------------------------------------------- #
# ChromaDB helpers
# --------------------------------------------------------------------------- #

def get_embedding_function():
    """Return the shared sentence-transformers embedding function."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )


def get_chroma_client(chroma_dir: str = CHROMA_DIR) -> chromadb.ClientAPI:
    """Create (or open) a persistent ChromaDB client."""
    os.makedirs(chroma_dir, exist_ok=True)
    return chromadb.PersistentClient(path=chroma_dir)


def get_or_create_collection(
    client: Optional[chromadb.ClientAPI] = None,
    chroma_dir: str = CHROMA_DIR,
):
    """Return the resume collection, creating it if necessary."""
    client = client or get_chroma_client(chroma_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )


def database_exists(chroma_dir: str = CHROMA_DIR) -> bool:
    """Return ``True`` if a populated collection already exists on disk."""
    if not os.path.isdir(chroma_dir):
        return False
    try:
        client = get_chroma_client(chroma_dir)
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=get_embedding_function(),
        )
        return collection.count() > 0
    except Exception:  # noqa: BLE001 - collection missing or unreadable
        return False


# --------------------------------------------------------------------------- #
# Public build API
# --------------------------------------------------------------------------- #

def build_vector_db(
    resume_dir: str = RESUME_DIR,
    chroma_dir: str = CHROMA_DIR,
    rebuild: bool = False,
    batch_size: int = 64,
) -> Dict[str, int]:
    """Build (or rebuild) the persistent resume vector database.

    Parameters
    ----------
    resume_dir:
        Directory containing resume PDFs (searched recursively).
    chroma_dir:
        Persistent ChromaDB directory.
    rebuild:
        When ``True`` the existing collection is dropped and rebuilt.
        When ``False`` and a populated database already exists, ingestion is
        skipped to avoid unnecessary recomputation.
    batch_size:
        Number of chunks embedded/added per ChromaDB call.

    Returns
    -------
    dict
        Summary counts: ``{"resumes": n, "chunks": m, "skipped": s}``.
    """
    client = get_chroma_client(chroma_dir)

    # Skip rebuild if a populated DB already exists and rebuild not forced.
    if not rebuild and database_exists(chroma_dir):
        collection = get_or_create_collection(client, chroma_dir)
        print(
            f"[INFO] Vector DB already exists with {collection.count()} chunks. "
            "Use rebuild=True to force a rebuild."
        )
        return {"resumes": 0, "chunks": collection.count(), "skipped": 1}

    # Fresh build: drop any existing collection first.
    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print("[INFO] Existing collection deleted for rebuild.")
        except Exception:  # noqa: BLE001 - nothing to delete
            pass

    collection = get_or_create_collection(client, chroma_dir)

    pdf_files = find_resume_files(resume_dir)
    if not pdf_files:
        print(f"[WARN] No PDF resumes found under '{resume_dir}'.")
        return {"resumes": 0, "chunks": 0, "skipped": 0}

    all_chunks: List[ResumeChunk] = []
    processed_resumes = 0

    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"[WARN] No extractable text in '{pdf_path}'. Skipping.")
            continue
        metadata = extract_metadata(text, pdf_path)
        resume_chunks = chunk_resume(text, metadata)
        if not resume_chunks:
            print(f"[WARN] No chunks produced for '{pdf_path}'. Skipping.")
            continue
        all_chunks.extend(resume_chunks)
        processed_resumes += 1
        print(
            f"[INFO] Processed '{metadata.candidate_name}' "
            f"({len(resume_chunks)} chunks) <- {pdf_path}"
        )

    if not all_chunks:
        print("[WARN] No chunks to index.")
        return {"resumes": processed_resumes, "chunks": 0, "skipped": 0}

    # Add chunks to ChromaDB in batches (embeddings computed automatically).
    for start in range(0, len(all_chunks), batch_size):
        batch = all_chunks[start : start + batch_size]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            metadatas=[
                {**c.metadata.to_chroma_metadata(), "section": c.section}
                for c in batch
            ],
        )

    print(
        f"[INFO] Indexed {len(all_chunks)} chunks from "
        f"{processed_resumes} resumes into '{chroma_dir}'."
    )
    return {
        "resumes": processed_resumes,
        "chunks": len(all_chunks),
        "skipped": 0,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the resume vector database (ChromaDB)."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a full rebuild even if the database already exists.",
    )
    parser.add_argument(
        "--resume-dir",
        default=RESUME_DIR,
        help=f"Directory containing resume PDFs (default: {RESUME_DIR}).",
    )
    args = parser.parse_args()

    summary = build_vector_db(resume_dir=args.resume_dir, rebuild=args.rebuild)
    print(f"[DONE] {summary}")

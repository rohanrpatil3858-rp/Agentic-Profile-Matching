"""
job_matcher.py
==============

Hybrid retrieval + LLM reasoning pipeline, reused from the "RAG based profile
matching v1" project (Milestone 2). Logic is unchanged; only the
``import resume_rag`` statement is switched to a package-relative import so it
works as ``tools.job_matcher`` inside this project.

Given a job description, this module:

1. Parses the job description into structured requirements
   (required skills, must-have skills, minimum experience).
2. Runs **semantic search** over the ChromaDB resume collection.
3. Runs **BM25 keyword search** focused on critical skills.
4. Fuses both rankings with **Reciprocal Rank Fusion (RRF)**.
5. Aggregates chunk-level hits into **candidate-level** rankings.
6. Applies **must-have filtering / penalties** (e.g. "5+ years Python").
7. Computes a transparent, weighted **0-100 match score**.
8. Uses an OpenAI-compatible LLM (via OpenRouter) to generate grounded
   reasoning for each top candidate, with a deterministic Python fallback
   when no API key is configured.
9. Emits the required JSON output.

Run from the command line::

    python -m tools.job_matcher --jd-file data/job_descriptions/ml_engineer.txt
    python -m tools.job_matcher --jd "Senior Python engineer with 5+ years ..."

The higher-level LangGraph agent (see ``matching_agent.py``) calls the
building-block functions in this module directly (``semantic_search``,
``bm25_search``, ``reciprocal_rank_fusion``, ``aggregate_candidates``,
``score_candidates``, ``generate_reasoning``) rather than the monolithic
``match_job`` helper, so that Parse JD / Search / Rank / Report are separate,
visible graph stages while still reusing this exact retrieval and scoring
logic.
"""

from __future__ import annotations

import os
import re
import json
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from . import resume_rag


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

load_dotenv()

# Number of candidates to return.
TOP_N_CANDIDATES = 10

# How many chunks to pull from each retriever. We over-fetch so that after
# aggregating to the candidate level we still comfortably cover the top 10.
SEMANTIC_TOP_K = 50
BM25_TOP_K = 50

# Default RRF constant.
DEFAULT_RRF_K = 60

# Transparent scoring weights (must sum to 1.0).
SCORE_WEIGHTS = {
    "semantic": 0.35,      # semantic relevance from fused retrieval
    "keyword": 0.25,       # critical / required skill coverage
    "experience": 0.20,    # experience requirement match
    "must_have": 0.20,     # must-have requirement satisfaction
}

# Penalty applied to a candidate's score when a must-have requirement fails.
MUST_HAVE_PENALTY = 0.5


# --------------------------------------------------------------------------- #
# Job description parsing
# --------------------------------------------------------------------------- #

@dataclass
class JobRequirements:
    """Structured requirements extracted from a job description."""

    raw_text: str
    required_skills: List[str] = field(default_factory=list)
    must_have_skills: List[str] = field(default_factory=list)
    min_experience: Optional[float] = None
    # Skill-specific experience demands, e.g. {"python": 5.0}.
    skill_experience: Dict[str, float] = field(default_factory=dict)


def _extract_min_experience(text: str) -> Optional[float]:
    """Find the minimum years of experience mentioned, if any."""
    years = []
    for match in re.finditer(r"(\d{1,2})\s*\+?\s*years?", text.lower()):
        try:
            years.append(float(match.group(1)))
        except ValueError:
            continue
    return max(years) if years else None


def _extract_skill_experience(text: str) -> Dict[str, float]:
    """Extract skill-specific experience demands like '5+ years Python'.

    Handles both orderings:
    * "5+ years of Python"
    * "Python (5+ years)"
    """
    lowered = text.lower()
    demands: Dict[str, float] = {}

    for skill in resume_rag.SKILL_VOCABULARY:
        # Word-boundary match (same as resume_rag._extract_skills); prevents
        # short skill names like "r", "c", "go" from matching inside larger
        # words (e.g. "r" inside "experience"). Still handles skills with
        # special characters such as "c++", "c#", "node.js".
        skill_pat = r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])"
        patterns = [
            rf"(\d{{1,2}})\s*\+?\s*years?(?:\s+of)?(?:\s+experience)?\s+(?:in\s+|with\s+)?{skill_pat}",
            rf"{skill_pat}[^.\n]{{0,30}}?(\d{{1,2}})\s*\+?\s*years?",
        ]
        for pat in patterns:
            m = re.search(pat, lowered)
            if m:
                try:
                    demands[skill] = max(demands.get(skill, 0.0), float(m.group(1)))
                except ValueError:
                    continue
    return demands


def _extract_must_have_skills(text: str, all_skills: List[str]) -> List[str]:
    """Identify skills flagged as mandatory in the job description.

    A skill is 'must-have' when it appears near strong requirement language
    (must, required, mandatory, essential) or within a "must have" block.
    """
    lowered = text.lower()
    must_have: List[str] = []

    # Lines / sentences that signal hard requirements.
    strong_markers = ["must", "required", "mandatory", "essential", "must-have",
                       "must have", "requirement"]

    # Sentence-level scan.
    sentences = re.split(r"[\n.;]", lowered)
    for sentence in sentences:
        if any(marker in sentence for marker in strong_markers):
            for skill in all_skills:
                pattern = r"(?<![A-Za-z0-9])" + re.escape(skill) + r"(?![A-Za-z0-9])"
                if re.search(pattern, sentence) and skill not in must_have:
                    must_have.append(skill)

    return must_have


def parse_job_description(text: str) -> JobRequirements:
    """Parse a free-text job description into :class:`JobRequirements`."""
    text = text.strip()
    # Reuse the resume skill extractor for consistency.
    required_skills = resume_rag._extract_skills(text)
    must_have_skills = _extract_must_have_skills(text, resume_rag.SKILL_VOCABULARY)
    min_experience = _extract_min_experience(text)
    skill_experience = _extract_skill_experience(text)

    return JobRequirements(
        raw_text=text,
        required_skills=required_skills,
        must_have_skills=must_have_skills,
        min_experience=min_experience,
        skill_experience=skill_experience,
    )


# --------------------------------------------------------------------------- #
# Retrieval: load all chunks + BM25 index
# --------------------------------------------------------------------------- #

@dataclass
class ChunkRecord:
    """In-memory representation of one indexed chunk."""

    chunk_id: str
    document: str
    metadata: Dict


def load_all_chunks(collection) -> List[ChunkRecord]:
    """Fetch every chunk (documents + metadata) from the ChromaDB collection."""
    data = collection.get(include=["documents", "metadatas"])
    records: List[ChunkRecord] = []
    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])
    for cid, doc, meta in zip(ids, docs, metas):
        records.append(ChunkRecord(chunk_id=cid, document=doc, metadata=meta or {}))
    return records


def _tokenize(text: str) -> List[str]:
    """Simple lowercase alphanumeric tokenizer for BM25."""
    return re.findall(r"[a-z0-9+#.]+", text.lower())


def semantic_search(collection, query: str, top_k: int = SEMANTIC_TOP_K) -> List[str]:
    """Return chunk IDs ranked by semantic similarity to the query."""
    result = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["metadatas", "distances"],
    )
    ids = result.get("ids", [[]])
    return ids[0] if ids else []


def bm25_search(
    records: List[ChunkRecord],
    query_terms: List[str],
    top_k: int = BM25_TOP_K,
) -> List[str]:
    """Rank chunks by BM25 over the provided critical query terms.

    Parameters
    ----------
    records:
        All chunk records (the BM25 corpus).
    query_terms:
        Critical skill / keyword tokens to score against.
    top_k:
        Maximum number of chunk IDs to return.
    """
    if not records or not query_terms:
        return []

    corpus_tokens = [_tokenize(r.document) for r in records]
    bm25 = BM25Okapi(corpus_tokens)

    # Flatten multi-word skills into tokens (e.g. "machine learning").
    tokens: List[str] = []
    for term in query_terms:
        tokens.extend(_tokenize(term))
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked = sorted(
        range(len(records)), key=lambda i: scores[i], reverse=True
    )
    # Keep only chunks with a positive score.
    ranked = [i for i in ranked if scores[i] > 0][:top_k]
    return [records[i].chunk_id for i in ranked]


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #

def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = DEFAULT_RRF_K,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked ID lists using Reciprocal Rank Fusion.

    RRF score for an item = sum over lists of ``1 / (k + rank)`` where ``rank``
    is 1-based. Items missing from a list simply contribute nothing for it.

    Returns a list of ``(chunk_id, fused_score)`` sorted descending.
    """
    fused: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            fused[chunk_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


# --------------------------------------------------------------------------- #
# Candidate-level aggregation & scoring
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    """Aggregated candidate with scoring inputs and final results."""

    candidate_name: str
    resume_path: str
    experience_years: float
    skills: List[str]
    education: str
    # Chunk excerpts (deduplicated, most relevant first).
    excerpts: List[str] = field(default_factory=list)
    # Sum of fused RRF scores across the candidate's chunks.
    semantic_score_raw: float = 0.0
    matched_skills: List[str] = field(default_factory=list)
    missing_must_have: List[str] = field(default_factory=list)
    match_score: int = 0
    reasoning: str = ""


def aggregate_candidates(
    fused: List[Tuple[str, float]],
    record_index: Dict[str, ChunkRecord],
) -> Dict[str, Candidate]:
    """Aggregate fused chunk scores into candidate-level records."""
    candidates: Dict[str, Candidate] = {}

    for chunk_id, score in fused:
        record = record_index.get(chunk_id)
        if record is None:
            continue
        meta = record.metadata
        path = meta.get("resume_path", "unknown")

        if path not in candidates:
            skills = [
                s.strip()
                for s in str(meta.get("skills", "")).split(",")
                if s.strip()
            ]
            candidates[path] = Candidate(
                candidate_name=meta.get("candidate_name", "Unknown"),
                resume_path=path,
                experience_years=float(meta.get("experience_years", 0.0) or 0.0),
                skills=skills,
                education=meta.get("education", ""),
            )

        candidate = candidates[path]
        candidate.semantic_score_raw += score
        # Store excerpt text (strip the leading "[Section]" tag for readability).
        excerpt = re.sub(r"^\[[^\]]+\]\s*", "", record.document).strip()
        if excerpt and excerpt not in candidate.excerpts:
            candidate.excerpts.append(excerpt)

    return candidates


def _skill_match(candidate: Candidate, requirements: JobRequirements) -> List[str]:
    """Return the required skills the candidate demonstrably has."""
    candidate_skills = {s.lower() for s in candidate.skills}
    matched = [
        skill
        for skill in requirements.required_skills
        if skill.lower() in candidate_skills
    ]
    return matched


def _evaluate_must_haves(
    candidate: Candidate, requirements: JobRequirements
) -> List[str]:
    """Return the list of must-have requirements the candidate fails.

    Covers must-have skills and skill-specific experience demands
    (e.g. "5+ years Python").
    """
    missing: List[str] = []
    candidate_skills = {s.lower() for s in candidate.skills}

    # Must-have skills the candidate lacks.
    for skill in requirements.must_have_skills:
        if skill.lower() not in candidate_skills:
            missing.append(skill)

    # Skill-specific experience (only enforce when the skill is must-have or
    # generally required). We approximate skill experience by the candidate's
    # total experience since per-skill tenure is rarely explicit.
    for skill, needed_years in requirements.skill_experience.items():
        has_skill = skill.lower() in candidate_skills
        if not has_skill:
            missing.append(f"{skill} ({needed_years:.0f}+ yrs)")
        elif candidate.experience_years + 1e-6 < needed_years:
            missing.append(f"{skill} {needed_years:.0f}+ yrs experience")

    # Global minimum experience.
    if (
        requirements.min_experience is not None
        and candidate.experience_years + 1e-6 < requirements.min_experience
    ):
        missing.append(f"{requirements.min_experience:.0f}+ yrs total experience")

    # De-duplicate preserving order.
    seen = set()
    unique = []
    for item in missing:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _normalize(value: float, max_value: float) -> float:
    """Scale ``value`` into [0, 1] given a batch maximum (safe for zeros)."""
    if max_value <= 0:
        return 0.0
    return min(1.0, value / max_value)


def score_candidates(
    candidates: Dict[str, Candidate],
    requirements: JobRequirements,
) -> List[Candidate]:
    """Compute transparent 0-100 match scores and rank candidates.

    The final score is a weighted blend of four normalised components:
    semantic relevance, keyword/skill coverage, experience match and
    must-have satisfaction, minus a penalty for each failed must-have.
    """
    if not candidates:
        return []

    max_semantic = max(c.semantic_score_raw for c in candidates.values()) or 1.0
    num_required = len(requirements.required_skills) or 1

    scored: List[Candidate] = []
    for candidate in candidates.values():
        # 1. Semantic relevance (normalised against the strongest candidate).
        semantic = _normalize(candidate.semantic_score_raw, max_semantic)

        # 2. Keyword / skill coverage.
        matched = _skill_match(candidate, requirements)
        candidate.matched_skills = matched
        keyword = len(matched) / num_required

        # 3. Experience match.
        if requirements.min_experience:
            experience = _normalize(
                candidate.experience_years, requirements.min_experience
            )
        else:
            # No explicit demand: reward more experience up to a soft cap.
            experience = _normalize(candidate.experience_years, 10.0)

        # 4. Must-have satisfaction.
        missing = _evaluate_must_haves(candidate, requirements)
        candidate.missing_must_have = missing
        total_must = (
            len(requirements.must_have_skills)
            + len(requirements.skill_experience)
            + (1 if requirements.min_experience else 0)
        )
        if total_must > 0:
            must_have = max(0.0, 1.0 - len(missing) / total_must)
        else:
            must_have = 1.0

        blended = (
            SCORE_WEIGHTS["semantic"] * semantic
            + SCORE_WEIGHTS["keyword"] * keyword
            + SCORE_WEIGHTS["experience"] * experience
            + SCORE_WEIGHTS["must_have"] * must_have
        )

        # Penalise (not eliminate) candidates who miss must-have requirements.
        if missing:
            blended *= MUST_HAVE_PENALTY

        candidate.match_score = int(round(max(0.0, min(1.0, blended)) * 100))
        # Keep only the most relevant excerpts.
        candidate.excerpts = candidate.excerpts[:4]
        scored.append(candidate)

    scored.sort(key=lambda c: c.match_score, reverse=True)
    return scored


# --------------------------------------------------------------------------- #
# LLM reasoning (OpenAI SDK / OpenRouter) + deterministic fallback
# --------------------------------------------------------------------------- #

def _get_llm_client():
    """Create an OpenAI-compatible client if an API key is configured.

    Returns ``(client, model)`` or ``(None, None)`` when unavailable.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() in {"", "your-api-key-here", "your_openrouter_api_key_here"}:
        return None, None

    try:
        from openai import OpenAI
    except ImportError:
        print("[WARN] openai package not installed; using fallback reasoning.")
        return None, None

    base_url = os.getenv("OPENAI_BASE_URL") or None
    model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        return client, model
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not initialise LLM client: {exc}")
        return None, None


def _fallback_reasoning(
    candidate: Candidate, requirements: JobRequirements
) -> str:
    """Deterministic, template-based reasoning used when no LLM is available.

    Grounded purely in extracted metadata and matched skills — never invents
    information.
    """
    parts: List[str] = []

    if candidate.matched_skills:
        parts.append(
            "Matches required skills: "
            + ", ".join(candidate.matched_skills)
            + "."
        )
    else:
        parts.append("No explicitly required skills were detected in the resume.")

    if candidate.experience_years:
        parts.append(
            f"Approximately {candidate.experience_years:.0f} years of experience."
        )

    if requirements.min_experience:
        if candidate.experience_years >= requirements.min_experience:
            parts.append(
                f"Meets the {requirements.min_experience:.0f}+ year experience bar."
            )
        else:
            parts.append(
                f"Falls short of the {requirements.min_experience:.0f}+ year "
                "experience requirement."
            )

    if candidate.missing_must_have:
        parts.append(
            "Gaps against must-have requirements: "
            + ", ".join(candidate.missing_must_have)
            + "."
        )
    else:
        parts.append("Satisfies all detected must-have requirements.")

    return " ".join(parts)


def _build_llm_prompt(
    candidate: Candidate, requirements: JobRequirements
) -> str:
    """Construct a grounded prompt that forbids hallucination."""
    excerpts = "\n".join(f"- {e}" for e in candidate.excerpts) or "- (none)"
    metadata = (
        f"Candidate: {candidate.candidate_name}\n"
        f"Experience (years): {candidate.experience_years:.0f}\n"
        f"Skills (extracted): {', '.join(candidate.skills) or 'none'}\n"
        f"Education: {candidate.education or 'not specified'}\n"
        f"Matched required skills: {', '.join(candidate.matched_skills) or 'none'}\n"
        f"Missing must-haves: {', '.join(candidate.missing_must_have) or 'none'}"
    )
    return (
        "You are a recruiting assistant. Using ONLY the job description, the "
        "retrieved resume excerpts, and the extracted metadata below, write 2-3 "
        "concise sentences explaining why this candidate matches. Do NOT invent "
        "skills or experience that are not present in the provided context. "
        "Explain which skills matched, relevant experience, and any important "
        "gaps.\n\n"
        f"JOB DESCRIPTION:\n{requirements.raw_text}\n\n"
        f"CANDIDATE METADATA:\n{metadata}\n\n"
        f"RETRIEVED RESUME EXCERPTS:\n{excerpts}\n\n"
        "REASONING:"
    )


def generate_reasoning(
    candidates: List[Candidate],
    requirements: JobRequirements,
) -> None:
    """Populate ``candidate.reasoning`` for each candidate (in place).

    Uses the configured LLM when available, otherwise the deterministic
    fallback. Any per-candidate LLM failure degrades gracefully to fallback.
    """
    client, model = _get_llm_client()

    for candidate in candidates:
        if client is None:
            candidate.reasoning = _fallback_reasoning(candidate, requirements)
            continue
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise recruiting assistant that never "
                            "fabricates information."
                        ),
                    },
                    {
                        "role": "user",
                        "content": _build_llm_prompt(candidate, requirements),
                    },
                ],
                temperature=0.2,
                max_tokens=200,
            )
            candidate.reasoning = response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            print(f"[WARN] LLM reasoning failed for {candidate.candidate_name}: {exc}")
            candidate.reasoning = _fallback_reasoning(candidate, requirements)


# --------------------------------------------------------------------------- #
# End-to-end matching pipeline
# --------------------------------------------------------------------------- #

def match_job(
    job_description: str,
    rrf_k: int = DEFAULT_RRF_K,
    top_n: int = TOP_N_CANDIDATES,
    use_llm: bool = True,
) -> Dict:
    """Run the full hybrid matching pipeline and return the JSON-ready result.

    Parameters
    ----------
    job_description:
        Raw job description text.
    rrf_k:
        RRF constant (default 60).
    top_n:
        Number of candidates to return.
    use_llm:
        When ``False`` skip LLM reasoning entirely (always use fallback).
    """
    requirements = parse_job_description(job_description)

    collection = resume_rag.get_or_create_collection()
    if collection.count() == 0:
        raise RuntimeError(
            "The vector database is empty. Build it first with "
            "`python -m tools.resume_rag --rebuild`."
        )

    records = load_all_chunks(collection)
    record_index = {r.chunk_id: r for r in records}

    # Critical terms for BM25 = must-have skills, else all required skills.
    critical_terms = requirements.must_have_skills or requirements.required_skills

    # 1. Semantic retrieval.
    semantic_ids = semantic_search(collection, job_description, SEMANTIC_TOP_K)
    # 2. BM25 keyword retrieval.
    bm25_ids = bm25_search(records, critical_terms, BM25_TOP_K)
    # 3. Hybrid fusion.
    fused = reciprocal_rank_fusion([semantic_ids, bm25_ids], k=rrf_k)

    # 4. Aggregate to candidate level and 5. score.
    candidates = aggregate_candidates(fused, record_index)
    ranked = score_candidates(candidates, requirements)
    top = ranked[:top_n]

    # 6. Reasoning.
    if use_llm:
        generate_reasoning(top, requirements)
    else:
        for candidate in top:
            candidate.reasoning = _fallback_reasoning(candidate, requirements)

    return {
        "job_description": job_description,
        "top_matches": [
            {
                "candidate_name": c.candidate_name,
                "resume_path": c.resume_path,
                "match_score": c.match_score,
                "matched_skills": c.matched_skills,
                "relevant_excerpts": c.excerpts,
                "reasoning": c.reasoning,
            }
            for c in top
        ],
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _read_job_description(args: argparse.Namespace) -> str:
    """Resolve the job description text from CLI arguments."""
    if args.jd_file:
        with open(args.jd_file, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    if args.jd:
        return args.jd.strip()
    raise SystemExit("Provide a job description via --jd-file or --jd.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match candidates to a job description using hybrid RAG."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--jd-file", help="Path to a job description text file.")
    source.add_argument("--jd", help="Job description text passed inline.")
    parser.add_argument(
        "--rrf-k", type=int, default=DEFAULT_RRF_K,
        help=f"RRF constant K (default: {DEFAULT_RRF_K}).",
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_CANDIDATES,
        help=f"Number of candidates to return (default: {TOP_N_CANDIDATES}).",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM reasoning and use the deterministic fallback.",
    )
    parser.add_argument(
        "--output", help="Optional path to write the JSON result to.",
    )
    args = parser.parse_args()

    job_description = _read_job_description(args)
    result = match_job(
        job_description,
        rrf_k=args.rrf_k,
        top_n=args.top_n,
        use_llm=not args.no_llm,
    )

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(output_json)
        print(f"[INFO] Result written to {args.output}")
    else:
        print(output_json)


if __name__ == "__main__":
    main()

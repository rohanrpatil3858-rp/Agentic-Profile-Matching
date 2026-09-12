"""tools/recruitment_tools.py

Adapter layer that exposes Milestone 1 (:mod:`tools.fs_tools`) and
Milestone 2 (:mod:`tools.resume_rag`, :mod:`tools.job_matcher`) functionality
as small, composable functions matching the shapes required by the Agentic
Profile Matching graph (:mod:`matching_agent`).

Design principle: reuse Milestone 2's retrieval/scoring building blocks
directly (``semantic_search``, ``bm25_search``, ``reciprocal_rank_fusion``,
``aggregate_candidates``, ``generate_reasoning``) instead of calling its
monolithic ``match_job()``, because ``match_job()`` re-derives requirements
from raw text and cannot honor externally-refined requirements (e.g. after
"Make TypeScript a must-have"). Milestone 2's code itself is not modified.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from . import job_matcher, resume_rag

# Re-exported for convenience / availability (Milestone 1 tools).
from .fs_tools import list_files, read_file, search_in_file, write_file  # noqa: F401

__all__ = [
    "ensure_vector_db",
    "extract_requirements",
    "refine_requirements",
    "search_resumes",
    "rank_candidates",
    "generate_report",
    "compare_candidates",
    "explain_ranking",
    "explain_ranking_change",
    "is_ranking_change_question",
    "generate_interview_questions",
    "classify_intent",
    "resolve_candidate_refs",
    "resolve_two_candidates",
    "resolve_candidate_ref",
    "list_files",
    "read_file",
    "search_in_file",
    "write_file",
]

TOP_N_DEFAULT = 10

# Simple, transparent ranking weights for Round 2 (must sum to 1.0).
SIMPLE_WEIGHTS = {"must_have": 0.6, "nice_to_have": 0.2, "experience": 0.2}

# Recommendation thresholds (0-100 score scale).
HIRE_THRESHOLD = 75
NO_HIRE_THRESHOLD = 45
BORDERLINE_LOW = 50
BORDERLINE_HIGH = 65

_SOFT_EXPERIENCE_CAP = 10.0


# --------------------------------------------------------------------------- #
# Vector DB lifecycle
# --------------------------------------------------------------------------- #

def ensure_vector_db(rebuild: bool = False) -> Dict[str, int]:
    """Build the resume vector DB if it does not already exist (or force it)."""
    return resume_rag.build_vector_db(rebuild=rebuild)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _slugify(text: str) -> str:
    """Turn arbitrary text into a short, URL/id-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "candidate"


def _requirements_dict_to_job_requirements(
    jd_text: str, requirements: Dict[str, Any]
) -> job_matcher.JobRequirements:
    """Build a Milestone-2 ``JobRequirements`` from the assignment's dict shape."""
    return job_matcher.JobRequirements(
        raw_text=jd_text or requirements.get("raw_text", ""),
        required_skills=list(requirements.get("relevant_skills", []) or []),
        must_have_skills=list(requirements.get("must_have", []) or []),
        min_experience=requirements.get("minimum_experience"),
        skill_experience={},
    )


def _candidate_to_dict(c: job_matcher.Candidate) -> Dict[str, Any]:
    """Convert a Milestone-2 ``Candidate`` into the plain-dict shape used here."""
    stem = c.resume_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    candidate_id = _slugify(stem or c.candidate_name)
    return {
        "candidate_id": candidate_id,
        "candidate_name": c.candidate_name,
        "resume_path": c.resume_path,
        "experience_years": c.experience_years,
        "skills": c.skills,
        "education": c.education,
        "excerpts": c.excerpts,
        "semantic_score_raw": c.semantic_score_raw,
    }


# --------------------------------------------------------------------------- #
# Round 0: requirement extraction / refinement
# --------------------------------------------------------------------------- #

def extract_requirements(jd: str) -> Dict[str, Any]:
    """Parse a job description into the assignment's structured requirements."""
    parsed = job_matcher.parse_job_description(jd)
    must_have = list(parsed.must_have_skills)
    nice_to_have = [s for s in parsed.required_skills if s not in must_have]
    return {
        "raw_text": jd,
        "must_have": must_have,
        "nice_to_have": nice_to_have,
        "minimum_experience": parsed.min_experience,
        "relevant_skills": list(parsed.required_skills),
    }


def refine_requirements(existing: Dict[str, Any], refinement_text: str) -> Dict[str, Any]:
    """Merge a natural-language refinement into existing requirements.

    Reuses Milestone 2's ``parse_job_description`` on just the refinement
    sentence (e.g. "Make TypeScript a must-have") so new must-have skills,
    additional relevant skills, and updated experience minimums are detected
    without any new NLP logic.
    """
    updated = {
        "raw_text": existing.get("raw_text", ""),
        "must_have": list(existing.get("must_have", []) or []),
        "nice_to_have": list(existing.get("nice_to_have", []) or []),
        "minimum_experience": existing.get("minimum_experience"),
        "relevant_skills": list(existing.get("relevant_skills", []) or []),
    }

    parsed = job_matcher.parse_job_description(refinement_text)

    for skill in parsed.must_have_skills:
        if skill not in updated["must_have"]:
            updated["must_have"].append(skill)
        if skill in updated["nice_to_have"]:
            updated["nice_to_have"].remove(skill)
        if skill not in updated["relevant_skills"]:
            updated["relevant_skills"].append(skill)

    for skill in parsed.required_skills:
        if skill not in updated["must_have"] and skill not in updated["nice_to_have"]:
            updated["nice_to_have"].append(skill)
        if skill not in updated["relevant_skills"]:
            updated["relevant_skills"].append(skill)

    if parsed.min_experience is not None:
        current = updated.get("minimum_experience")
        updated["minimum_experience"] = (
            max(current, parsed.min_experience) if current else parsed.min_experience
        )

    return updated


# --------------------------------------------------------------------------- #
# Round 1: search (retrieval only, no scoring yet)
# --------------------------------------------------------------------------- #

def search_resumes(
    job_description: str,
    requirements: Dict[str, Any],
    top_n: int = TOP_N_DEFAULT,
) -> List[Dict[str, Any]]:
    """Retrieve a candidate pool via hybrid (semantic + BM25) search.

    This is Round 1 of the assignment's pipeline: retrieval and fusion only.
    Scoring against must-have / nice-to-have requirements happens in
    :func:`rank_candidates` (Round 2).
    """
    ensure_vector_db()
    collection = resume_rag.get_or_create_collection()
    if collection.count() == 0:
        return []

    records = job_matcher.load_all_chunks(collection)
    record_index = {r.chunk_id: r for r in records}

    jr = _requirements_dict_to_job_requirements(job_description, requirements)
    critical_terms = jr.must_have_skills or jr.required_skills

    query_text = job_description or " ".join(jr.required_skills) or "software engineer"
    semantic_ids = job_matcher.semantic_search(collection, query_text, job_matcher.SEMANTIC_TOP_K)
    bm25_ids = job_matcher.bm25_search(records, critical_terms, job_matcher.BM25_TOP_K)
    fused = job_matcher.reciprocal_rank_fusion([semantic_ids, bm25_ids])

    candidates = job_matcher.aggregate_candidates(fused, record_index)
    pool = sorted(candidates.values(), key=lambda c: c.semantic_score_raw, reverse=True)
    pool = pool[:top_n]

    return [_candidate_to_dict(c) for c in pool]


# --------------------------------------------------------------------------- #
# Round 2: ranking against must-have / nice-to-have requirements
# --------------------------------------------------------------------------- #

def _match_skills(candidate_skills: List[str], wanted: List[str]) -> Tuple[List[str], List[str]]:
    """Split ``wanted`` skills into (matched, missing) against a candidate."""
    have = {s.lower() for s in candidate_skills}
    matched = [s for s in wanted if s.lower() in have]
    missing = [s for s in wanted if s.lower() not in have]
    return matched, missing


def rank_candidates(
    pool: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Score and rank a candidate pool against structured requirements.

    Round 2 of the pipeline. Uses a simple, transparent weighted formula
    (must-have 60% / nice-to-have 20% / experience 20%, with a small RAG
    relevance bonus) then calls Milestone 2's ``generate_reasoning`` for
    grounded, LLM-or-fallback explanations.

    Returns
    -------
    tuple
        ``(scored_candidates, candidate_reasoning)`` where
        ``candidate_reasoning`` is keyed by ``candidate_id``.
    """
    must_have = list(requirements.get("must_have", []) or [])
    nice_to_have = list(requirements.get("nice_to_have", []) or [])
    min_experience = requirements.get("minimum_experience")

    max_semantic = max((c.get("semantic_score_raw", 0.0) for c in pool), default=0.0) or 1.0

    scored: List[Dict[str, Any]] = []
    for c in pool:
        skills = c.get("skills", [])
        matched_must, missing_must = _match_skills(skills, must_have)
        matched_nice, missing_nice = _match_skills(skills, nice_to_have)

        must_have_score = (len(matched_must) / len(must_have)) if must_have else 1.0
        nice_to_have_score = (len(matched_nice) / len(nice_to_have)) if nice_to_have else 1.0

        experience_years = c.get("experience_years", 0.0) or 0.0
        cap = min_experience if min_experience else _SOFT_EXPERIENCE_CAP
        experience_score = min(1.0, experience_years / cap) if cap else 0.0

        blended = (
            SIMPLE_WEIGHTS["must_have"] * must_have_score
            + SIMPLE_WEIGHTS["nice_to_have"] * nice_to_have_score
            + SIMPLE_WEIGHTS["experience"] * experience_score
        )
        relevance_bonus = (c.get("semantic_score_raw", 0.0) / max_semantic) * 0.05
        blended = max(0.0, min(1.0, blended + relevance_bonus))
        score = int(round(blended * 100))

        meets_experience = bool(min_experience) and experience_years >= min_experience

        strengths: List[str] = []
        if matched_must:
            strengths.append("Has must-have skills: " + ", ".join(matched_must))
        if matched_nice:
            strengths.append("Has nice-to-have skills: " + ", ".join(matched_nice))
        if min_experience and experience_years >= min_experience:
            strengths.append(f"Meets the {min_experience:.0f}+ year experience requirement")

        gaps: List[str] = []
        if missing_must:
            gaps.append("Missing must-have skills: " + ", ".join(missing_must))
        if missing_nice:
            gaps.append("Missing nice-to-have skills: " + ", ".join(missing_nice))
        if min_experience and experience_years < min_experience:
            gaps.append(
                f"Has {experience_years:.0f} years, below the "
                f"{min_experience:.0f}+ year requirement"
            )

        if score >= HIRE_THRESHOLD and not missing_must:
            recommendation = "Hire"
        elif score <= NO_HIRE_THRESHOLD or missing_must:
            recommendation = "No Hire" if score <= NO_HIRE_THRESHOLD else "Consider"
        else:
            recommendation = "Consider"

        improvement_suggestions = None
        if BORDERLINE_LOW <= score <= BORDERLINE_HIGH:
            suggestion_bits = []
            if missing_must:
                suggestion_bits.append(
                    "closing the gap on: " + ", ".join(missing_must)
                )
            if missing_nice:
                suggestion_bits.append(
                    "gaining exposure to: " + ", ".join(missing_nice)
                )
            if suggestion_bits:
                improvement_suggestions = (
                    "Could become a stronger fit by " + "; and ".join(suggestion_bits) + "."
                )

        scored.append(
            {
                **c,
                "matched_must_have": matched_must,
                "missing_must_have": missing_must,
                "matched_nice_to_have": matched_nice,
                "missing_nice_to_have": missing_nice,
                "meets_experience": meets_experience,
                "score": score,
                "strengths": strengths,
                "gaps": gaps,
                "recommendation": recommendation,
                "improvement_suggestions": improvement_suggestions,
            }
        )

    scored.sort(key=lambda c: c["score"], reverse=True)

    # Batch-generate grounded reasoning via Milestone 2's (unmodified) LLM path.
    jr = _requirements_dict_to_job_requirements(requirements.get("raw_text", ""), requirements)
    pseudo_candidates = []
    for c in scored:
        pseudo = job_matcher.Candidate(
            candidate_name=c["candidate_name"],
            resume_path=c["resume_path"],
            experience_years=c.get("experience_years", 0.0),
            skills=c.get("skills", []),
            education=c.get("education", ""),
            excerpts=c.get("excerpts", []),
        )
        pseudo.matched_skills = c["matched_must_have"] + c["matched_nice_to_have"]
        pseudo.missing_must_have = c["missing_must_have"]
        pseudo_candidates.append(pseudo)

    if pseudo_candidates:
        job_matcher.generate_reasoning(pseudo_candidates, jr)

    candidate_reasoning: Dict[str, Any] = {}
    for c, pseudo in zip(scored, pseudo_candidates):
        c["reasoning"] = pseudo.reasoning
        candidate_reasoning[c["candidate_id"]] = {
            "score": c["score"],
            "matched_must_have": c["matched_must_have"],
            "missing_must_have": c["missing_must_have"],
            "matched_nice_to_have": c["matched_nice_to_have"],
            "missing_nice_to_have": c["missing_nice_to_have"],
            "strengths": c["strengths"],
            "gaps": c["gaps"],
            "reasoning": c["reasoning"],
            "recommendation": c["recommendation"],
            "improvement_suggestions": c["improvement_suggestions"],
        }

    return scored, candidate_reasoning


# --------------------------------------------------------------------------- #
# Round 3: reporting
# --------------------------------------------------------------------------- #

def generate_report(candidates: List[Dict[str, Any]], requirements: Dict[str, Any]) -> str:
    """Render a markdown shortlist report."""
    lines = ["# Candidate Shortlist Report", ""]
    lines.append(f"**Must-have skills:** {', '.join(requirements.get('must_have', [])) or 'none specified'}")
    lines.append(f"**Nice-to-have skills:** {', '.join(requirements.get('nice_to_have', [])) or 'none specified'}")
    min_exp = requirements.get("minimum_experience")
    lines.append(f"**Minimum experience:** {f'{min_exp:.0f}+ years' if min_exp else 'not specified'}")
    lines.append("")

    if not candidates:
        lines.append(
            "No matching candidates were found. Try broadening the requirements "
            "or check that resumes have been indexed."
        )
        return "\n".join(lines)

    for i, c in enumerate(candidates, start=1):
        lines.append(f"## {i}. {c['candidate_name']} — Score: {c['score']}/100 ({c['recommendation']})")
        lines.append(f"- **Candidate ID:** `{c['candidate_id']}`")
        lines.append(f"- **Experience:** {c.get('experience_years', 0):.0f} years")
        lines.append(f"- **Matched must-have:** {', '.join(c['matched_must_have']) or 'none'}")
        if c["missing_must_have"]:
            lines.append(f"- **Missing must-have:** {', '.join(c['missing_must_have'])}")
        lines.append(f"- **Matched nice-to-have:** {', '.join(c['matched_nice_to_have']) or 'none'}")
        for s in c.get("strengths", []):
            lines.append(f"- ✅ {s}")
        for g in c.get("gaps", []):
            lines.append(f"- ⚠️ {g}")
        lines.append(f"- **Reasoning:** {c.get('reasoning', '')}")
        if c.get("improvement_suggestions"):
            lines.append(f"- **Improvement suggestion:** {c['improvement_suggestions']}")
        lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Round 4: comparison / explanation / interview questions
# --------------------------------------------------------------------------- #

def compare_candidates(
    candidate_ids: List[str],
    shortlist: List[Dict[str, Any]],
    reasoning: Dict[str, Any],
) -> str:
    """Render a markdown head-to-head comparison for the given candidate IDs."""
    by_id = {c["candidate_id"]: c for c in shortlist}
    found = [cid for cid in candidate_ids if cid in by_id]

    if not found:
        return "I couldn't find those candidates in the current shortlist. Try running a search first."

    lines = ["# Candidate Comparison", ""]
    for cid in found:
        c = by_id[cid]
        r = reasoning.get(cid, {})
        lines.append(f"## {c['candidate_name']} (`{cid}`)")
        lines.append(f"- **Score:** {c.get('score', 'n/a')}/100 ({c.get('recommendation', 'n/a')})")
        lines.append(f"- **Experience:** {c.get('experience_years', 0):.0f} years")
        lines.append(f"- **Matched must-have:** {', '.join(c.get('matched_must_have', [])) or 'none'}")
        lines.append(f"- **Missing must-have:** {', '.join(c.get('missing_must_have', [])) or 'none'}")
        for s in r.get("strengths", []):
            lines.append(f"- ✅ {s}")
        for g in r.get("gaps", []):
            lines.append(f"- ⚠️ {g}")
        lines.append("")

    return "\n".join(lines)


def explain_ranking(
    a_id: Optional[str],
    b_id: Optional[str],
    shortlist: List[Dict[str, Any]],
    reasoning: Dict[str, Any],
) -> str:
    """Explain why candidate ``a_id`` ranks higher/lower than ``b_id``."""
    by_id = {c["candidate_id"]: c for c in shortlist}
    if not a_id or not b_id or a_id not in by_id or b_id not in by_id:
        return "I couldn't find both candidates to compare. Try running a search or comparison first."

    a, b = by_id[a_id], by_id[b_id]
    higher, lower = (a, b) if a.get("score", 0) >= b.get("score", 0) else (b, a)

    if higher.get("score", 0) == lower.get("score", 0):
        return (
            f"**{a['candidate_name']}** and **{b['candidate_name']}** are effectively "
            f"tied at {a.get('score')}/100 — both satisfy the same must-have and "
            "nice-to-have requirements, so ordering between them is interchangeable. "
            f"({a['candidate_name']}: {a.get('experience_years', 0):.0f} yrs, "
            f"{b['candidate_name']}: {b.get('experience_years', 0):.0f} yrs experience.)"
        )

    higher_must = set(s.lower() for s in higher.get("matched_must_have", []))
    lower_must = set(s.lower() for s in lower.get("matched_must_have", []))
    extra_must = higher_must - lower_must

    higher_nice = set(s.lower() for s in higher.get("matched_nice_to_have", []))
    lower_nice = set(s.lower() for s in lower.get("matched_nice_to_have", []))
    extra_nice = higher_nice - lower_nice

    exp_diff = (higher.get("experience_years", 0) or 0) - (lower.get("experience_years", 0) or 0)

    lines = [f"**{higher['candidate_name']}** ranks higher than **{lower['candidate_name']}** because:"]
    if extra_must:
        lines.append(f"- Has additional must-have skills: {', '.join(sorted(extra_must))}")
    if extra_nice:
        lines.append(f"- Has additional nice-to-have skills: {', '.join(sorted(extra_nice))}")
    if abs(exp_diff) > 0.01:
        direction = "more" if exp_diff > 0 else "less"
        lines.append(f"- Has {abs(exp_diff):.0f} years {direction} experience")
    if lower.get("missing_must_have"):
        lines.append(
            f"- {lower['candidate_name']} is missing: {', '.join(lower['missing_must_have'])}"
        )
    lines.append(
        f"- Final scores: {higher['candidate_name']} = {higher.get('score')}/100, "
        f"{lower['candidate_name']} = {lower.get('score')}/100"
    )

    return "\n".join(lines)


def is_ranking_change_question(message: str) -> bool:
    """Return True if the message asks about a candidate's movement over time."""
    lowered = message.lower()
    return any(
        phrase in lowered
        for phrase in [
            "move", "moved", "jump", "jumped", "went from", "from #",
            "rose", "climb", "climbed", "drop", "dropped", "fell", "fall",
        ]
    )


def explain_ranking_change(
    candidate_id: Optional[str],
    shortlist: List[Dict[str, Any]],
    previous_ranking: List[Dict[str, Any]],
    reasoning: Dict[str, Any],
) -> str:
    """Explain how a candidate's rank changed vs. the previous ranking snapshot.

    Uses the ``previous_ranking`` captured before the last re-rank (positions +
    scores) to describe movement (e.g. "#3 -> #1") and grounds the explanation
    in the candidate's currently matched/missing requirements.
    """
    by_id = {c["candidate_id"]: c for c in shortlist}
    if not candidate_id or candidate_id not in by_id:
        return "I couldn't find that candidate in the current shortlist. Try running a search first."

    current = by_id[candidate_id]
    name = current["candidate_name"]
    current_rank = next(
        (i + 1 for i, c in enumerate(shortlist) if c["candidate_id"] == candidate_id),
        None,
    )
    current_score = current.get("score")

    prev = next(
        (p for p in previous_ranking if p["candidate_id"] == candidate_id), None
    )
    if prev is None:
        return (
            f"**{name}** is now ranked #{current_rank} (score {current_score}/100). "
            "There is no earlier ranking on record for this candidate, so there is "
            "no position change to explain yet."
        )

    prev_rank = prev.get("rank")
    prev_score = prev.get("score")
    r = reasoning.get(candidate_id, {})

    if prev_rank == current_rank:
        header = (
            f"**{name}** stayed at #{current_rank} "
            f"(score {prev_score}/100 -> {current_score}/100)."
        )
    else:
        direction = "up" if current_rank < prev_rank else "down"
        header = (
            f"**{name}** moved {direction} from #{prev_rank} to #{current_rank} "
            f"(score {prev_score}/100 -> {current_score}/100) after the requirements changed:"
        )

    lines = [header]
    if r.get("matched_must_have"):
        lines.append(f"- Now satisfies must-have skills: {', '.join(r['matched_must_have'])}")
    if r.get("missing_must_have"):
        lines.append(f"- Still missing must-have skills: {', '.join(r['missing_must_have'])}")
    if r.get("matched_nice_to_have"):
        lines.append(f"- Matches nice-to-have skills: {', '.join(r['matched_nice_to_have'])}")
    lines.append(
        "- The change reflects the updated job requirements applied during re-ranking."
    )

    return "\n".join(lines)


def generate_interview_questions(
    candidate_id: Optional[str],
    shortlist: List[Dict[str, Any]],
    requirements: Dict[str, Any],
) -> str:
    """Generate ~5 targeted interview questions for a candidate.

    Tries an LLM call (via Milestone 2's shared client helper) grounded in the
    candidate's matched/missing skills; falls back to a deterministic
    template-based list when no LLM is configured or on any failure.
    """
    by_id = {c["candidate_id"]: c for c in shortlist}
    if not candidate_id or candidate_id not in by_id:
        return "I couldn't find that candidate in the current shortlist."

    c = by_id[candidate_id]
    matched = c.get("matched_must_have", []) + c.get("matched_nice_to_have", [])
    missing = c.get("missing_must_have", []) + c.get("missing_nice_to_have", [])

    client, model = job_matcher._get_llm_client()
    if client is not None:
        try:
            prompt = (
                "Generate exactly 5 targeted technical interview questions for a "
                f"candidate named {c['candidate_name']} applying to a role requiring "
                f"must-have skills: {', '.join(requirements.get('must_have', [])) or 'none'} "
                f"and nice-to-have skills: {', '.join(requirements.get('nice_to_have', [])) or 'none'}.\n"
                f"The candidate's matched skills are: {', '.join(matched) or 'none'}.\n"
                f"The candidate's skill gaps are: {', '.join(missing) or 'none'}.\n"
                "Base questions on these matched skills (to verify depth) and gaps "
                "(to assess adaptability). Return only a numbered list of 5 questions."
            )
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise technical interviewer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )
            text = response.choices[0].message.content.strip()
            return f"# Interview Questions for {c['candidate_name']}\n\n{text}"
        except Exception:  # noqa: BLE001 - fall through to deterministic fallback
            pass

    questions = []
    for skill in matched[:3]:
        questions.append(f"Can you walk me through a project where you used {skill} in depth?")
    for skill in missing[:2]:
        questions.append(f"You don't list {skill} explicitly — how would you ramp up on it quickly?")

    fillers = [
        "Tell me about a challenging technical problem you solved recently.",
        "How do you approach code reviews and ensuring code quality on your team?",
        "Describe a time you had to learn a new technology quickly for a project.",
        "How do you balance delivery speed against technical debt?",
        "Tell me about a disagreement with a teammate and how you resolved it.",
    ]
    for filler in fillers:
        if len(questions) >= 5:
            break
        questions.append(filler)

    lines = [f"# Interview Questions for {c['candidate_name']}", ""]
    for i, q in enumerate(questions[:5], start=1):
        lines.append(f"{i}. {q}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Conversational intent classification & candidate resolution
# --------------------------------------------------------------------------- #

def classify_intent(message: str, has_shortlist: bool) -> str:
    """Heuristically classify a user message into a graph route.

    Returns one of: ``"interview"``, ``"compare"``, ``"explain"``,
    ``"refine"``, ``"new_search"``.
    """
    lowered = message.lower()

    if "interview question" in lowered or "screening question" in lowered:
        return "interview"
    if "compare" in lowered:
        return "compare"
    if any(
        phrase in lowered
        for phrase in [
            "why did", "why does", "rank higher", "ranked higher",
            "moved from", "explain the ranking", "why is",
        ]
    ):
        return "explain"
    if has_shortlist and any(
        phrase in lowered
        for phrase in [
            "must-have", "must have", "nice-to-have", "nice to have",
            "prioritize", "minimum experience", "at least", "rerank", "re-rank",
        ]
    ):
        return "refine"
    return "new_search"


def resolve_candidate_refs(message: str, shortlist: List[Dict[str, Any]], max_n: int = 3) -> List[str]:
    """Resolve one or more candidate references from free text.

    Supports ordinal phrasing ("top candidate", "top 3"), first-name
    substring matching, and falls back to the top ``max_n`` candidates.
    """
    if not shortlist:
        return []

    lowered = message.lower()

    match = re.search(r"top\s+(\d+)", lowered)
    if match:
        n = min(int(match.group(1)), len(shortlist), max_n if max_n else len(shortlist))
        return [c["candidate_id"] for c in shortlist[:n]]

    matched_ids: List[str] = []
    for c in shortlist:
        first_name = c["candidate_name"].split()[0].lower()
        if first_name in lowered:
            matched_ids.append(c["candidate_id"])
    if matched_ids:
        return matched_ids[:max_n]

    return [c["candidate_id"] for c in shortlist[: min(max_n, len(shortlist))]]


def resolve_two_candidates(
    message: str, shortlist: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve exactly two candidate IDs for comparison/explanation."""
    if not shortlist:
        return None, None

    refs = resolve_candidate_refs(message, shortlist, max_n=2)
    if len(refs) >= 2:
        return refs[0], refs[1]
    if len(shortlist) >= 2:
        return shortlist[0]["candidate_id"], shortlist[1]["candidate_id"]
    if shortlist:
        return shortlist[0]["candidate_id"], None
    return None, None


def resolve_candidate_ref(message: str, shortlist: List[Dict[str, Any]]) -> Optional[str]:
    """Resolve a single candidate reference (e.g. for interview questions)."""
    if not shortlist:
        return None

    lowered = message.lower()
    if any(p in lowered for p in ["top candidate", "top match", "first candidate"]):
        return shortlist[0]["candidate_id"]

    for c in shortlist:
        first_name = c["candidate_name"].split()[0].lower()
        if first_name in lowered:
            return c["candidate_id"]

    return shortlist[0]["candidate_id"]


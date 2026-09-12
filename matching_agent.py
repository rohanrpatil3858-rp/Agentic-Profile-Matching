"""matching_agent.py

LangGraph orchestration for the Agentic Profile Matching assistant.

Graph overview
--------------
```
START -> classify_intent --(route_intent)--> parse_jd | extract_requirements
                                              | compare_candidates
                                              | explain_ranking
                                              | interview_questions

parse_jd -> extract_requirements -> search_resumes -> rank_candidates
         -> generate_report -> human_feedback -> END

compare_candidates | explain_ranking | interview_questions -> human_feedback -> END
```

Conversation state (job description, requirements, shortlist, reasoning,
history) is carried in :class:`AgentState` and persisted across turns via a
LangGraph ``MemorySaver`` checkpointer keyed by ``thread_id`` -- this is what
allows a user to refine requirements, compare candidates, ask "why", and
request interview questions, all within a single ongoing conversation.
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from tools import recruitment_tools as rt


class AgentState(TypedDict, total=False):
    """Conversation + pipeline state carried across graph nodes and turns."""

    conversation_history: List[Dict[str, str]]
    job_description: str
    job_requirements: Dict[str, Any]
    candidate_shortlist: List[Dict[str, Any]]
    candidate_reasoning: Dict[str, Any]
    previous_ranking: List[Dict[str, Any]]
    last_intent: str
    last_response: str
    round: int
    user_message: str


# --------------------------------------------------------------------------- #
# Node functions
# --------------------------------------------------------------------------- #

def classify_intent(state: AgentState) -> Dict[str, Any]:
    """Route the incoming user message based on lightweight intent heuristics."""
    message = state.get("user_message", "")
    has_shortlist = bool(state.get("candidate_shortlist"))
    intent = rt.classify_intent(message, has_shortlist)
    return {"last_intent": intent}


def parse_jd_node(state: AgentState) -> Dict[str, Any]:
    """Treat the latest user message as a (new) job description."""
    return {"job_description": state.get("user_message", "")}


def extract_requirements_node(state: AgentState) -> Dict[str, Any]:
    """Extract or refine structured requirements from the job description."""
    intent = state.get("last_intent")
    existing = state.get("job_requirements")
    message = state.get("user_message", "")

    if intent == "refine" and existing:
        updated = rt.refine_requirements(existing, message)
    else:
        jd = state.get("job_description") or message
        updated = rt.extract_requirements(jd)

    return {"job_requirements": updated}


def search_resumes_node(state: AgentState) -> Dict[str, Any]:
    """Round 1: retrieve a candidate pool via hybrid search.

    Before overwriting the shortlist, snapshot the previous ranking (candidate
    positions + scores) so that a later "why did X move from #3 to #1?" query
    can be answered after a requirement refinement.
    """
    old_shortlist = state.get("candidate_shortlist", []) or []
    previous_ranking = [
        {
            "candidate_id": c["candidate_id"],
            "candidate_name": c["candidate_name"],
            "rank": i + 1,
            "score": c.get("score"),
        }
        for i, c in enumerate(old_shortlist)
    ]
    jd = state.get("job_description", "")
    requirements = state.get("job_requirements", {}) or {}
    pool = rt.search_resumes(jd, requirements, top_n=10)
    return {
        "candidate_shortlist": pool,
        "round": 1,
        "previous_ranking": previous_ranking,
    }


def rank_candidates_node(state: AgentState) -> Dict[str, Any]:
    """Round 2: score and rank the candidate pool."""
    pool = state.get("candidate_shortlist", []) or []
    requirements = state.get("job_requirements", {}) or {}
    ranked, reasoning = rt.rank_candidates(pool, requirements)
    return {"candidate_shortlist": ranked, "candidate_reasoning": reasoning, "round": 2}


def generate_report_node(state: AgentState) -> Dict[str, Any]:
    """Round 3: render the final shortlist report."""
    shortlist = state.get("candidate_shortlist", []) or []
    requirements = state.get("job_requirements", {}) or {}
    report = rt.generate_report(shortlist, requirements)
    return {"last_response": report, "round": 3}


def compare_candidates_node(state: AgentState) -> Dict[str, Any]:
    """Compare two or more candidates from the current shortlist."""
    message = state.get("user_message", "")
    shortlist = state.get("candidate_shortlist", []) or []
    reasoning = state.get("candidate_reasoning", {}) or {}
    candidate_ids = rt.resolve_candidate_refs(message, shortlist, max_n=3)
    response = rt.compare_candidates(candidate_ids, shortlist, reasoning)
    return {"last_response": response}


def explain_ranking_node(state: AgentState) -> Dict[str, Any]:
    """Explain a candidate's ranking.

    Handles two flavours of "why" question:
    * movement over time ("why did John move from #3 to #1?") -- compared
      against the ``previous_ranking`` snapshot; and
    * head-to-head ("why did the top candidate rank higher than the second?").
    """
    message = state.get("user_message", "")
    shortlist = state.get("candidate_shortlist", []) or []
    reasoning = state.get("candidate_reasoning", {}) or {}
    previous_ranking = state.get("previous_ranking", []) or []

    refs = rt.resolve_candidate_refs(message, shortlist, max_n=2)
    if previous_ranking and (rt.is_ranking_change_question(message) or len(refs) == 1):
        candidate_id = refs[0] if refs else (
            shortlist[0]["candidate_id"] if shortlist else None
        )
        response = rt.explain_ranking_change(
            candidate_id, shortlist, previous_ranking, reasoning
        )
    else:
        a_id, b_id = rt.resolve_two_candidates(message, shortlist)
        response = rt.explain_ranking(a_id, b_id, shortlist, reasoning)
    return {"last_response": response}


def interview_questions_node(state: AgentState) -> Dict[str, Any]:
    """Generate interview questions for a referenced candidate."""
    message = state.get("user_message", "")
    shortlist = state.get("candidate_shortlist", []) or []
    requirements = state.get("job_requirements", {}) or {}
    candidate_id = rt.resolve_candidate_ref(message, shortlist)
    response = rt.generate_interview_questions(candidate_id, shortlist, requirements)
    return {"last_response": response}


def human_feedback_node(state: AgentState) -> Dict[str, Any]:
    """Append the latest user/assistant turn to the conversation history."""
    history = list(state.get("conversation_history", []) or [])
    history.append({"role": "user", "content": state.get("user_message", "")})
    history.append({"role": "assistant", "content": state.get("last_response", "")})
    return {"conversation_history": history}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #

_ROUTES = {
    "new_search": "parse_jd",
    "refine": "extract_requirements",
    "compare": "compare_candidates",
    "explain": "explain_ranking",
    "interview": "interview_questions",
}


def route_intent(state: AgentState) -> str:
    """Map the classified intent to the next graph node."""
    return _ROUTES.get(state.get("last_intent", "new_search"), "parse_jd")


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #

def build_graph() -> StateGraph:
    """Construct (but do not compile) the Agentic Profile Matching graph."""
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("compare_candidates", compare_candidates_node)
    graph.add_node("explain_ranking", explain_ranking_node)
    graph.add_node("interview_questions", interview_questions_node)
    graph.add_node("human_feedback", human_feedback_node)

    graph.add_edge(START, "classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "parse_jd": "parse_jd",
            "extract_requirements": "extract_requirements",
            "compare_candidates": "compare_candidates",
            "explain_ranking": "explain_ranking",
            "interview_questions": "interview_questions",
        },
    )

    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", "human_feedback")

    graph.add_edge("compare_candidates", "human_feedback")
    graph.add_edge("explain_ranking", "human_feedback")
    graph.add_edge("interview_questions", "human_feedback")

    graph.add_edge("human_feedback", END)

    return graph


_graph = build_graph()
agent = _graph.compile(checkpointer=MemorySaver())


def chat(message: str, thread_id: str = "default") -> str:
    """Send a single message to the agent and return its text reply.

    Conversation state (job description, requirements, shortlist, reasoning)
    persists across calls that share the same ``thread_id``.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"user_message": message}, config=config)
    return result.get("last_response", "")


if __name__ == "__main__":
    sample_jd = (
        "We are hiring a Senior Backend Engineer. Must have Python and "
        "5+ years of experience. Nice to have: Docker and AWS."
    )
    reply = chat(sample_jd, thread_id="smoke-test")
    print(reply)

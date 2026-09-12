# Agentic Profile Matching

A conversational, LangGraph-powered recruiting assistant. Describe a role in
plain English and the agent searches a resume corpus, ranks candidates,
explains its reasoning, compares candidates head-to-head, refines requirements
mid-conversation, and generates interview questions — all through a simple
Gradio chat interface.

This project is the **final assignment** in a three-part series and reuses the
earlier milestones internally (their code is bundled here, so this repo is
fully self-contained — you do **not** need the original milestone folders):

- **Milestone 1 — File-system tools** (`tools/fs_tools.py`): read / list /
  search / write document files (PDF, DOCX, TXT), sandboxed to the project.
- **Milestone 2 — RAG retrieval** (`tools/resume_rag.py`, `tools/job_matcher.py`):
  ChromaDB + `sentence-transformers` embeddings, hybrid semantic + BM25
  retrieval fused with Reciprocal Rank Fusion, transparent scoring, and
  LLM-generated reasoning.

---

## Features

**Part A — LangGraph agent**

```
START -> classify_intent --> parse_jd -> extract_requirements -> search_resumes
                                      -> rank_candidates -> generate_report
                                      -> human_feedback -> END
```

`AgentState` carries `conversation_history`, `job_description`,
`job_requirements`, `candidate_shortlist`, `candidate_reasoning`, and a
`previous_ranking` snapshot (used to explain how a candidate's rank changed
after requirements are refined). State is persisted across chat turns with a
LangGraph `MemorySaver` checkpointer, which is what makes iterative refinement
and follow-up questions work.

**Part B — Conversational interaction & iterative refinement**

- Natural-language search: *"Find candidates with React and 3+ years experience."*
- Head-to-head comparison: *"Compare the top 3 candidates."*
- Ranking explanations: *"Why did the top candidate rank higher than the second?"*
- Live requirement changes: *"Make TypeScript a must-have and rerank."*

**Part C — Multi-round screening & explainability**

- Round 1: retrieve the top candidates from the corpus.
- Round 2: deeper scoring and analysis.
- Round 3: hire / no-hire recommendation per candidate.
- Strengths, gaps, matched/missing requirements, reasoning, and improvement
  suggestions for borderline candidates.

**Required tools**

- `extract_requirements(jd)` → `{must_have, nice_to_have, minimum_experience, relevant_skills}`
- `compare_candidates(candidate_ids)` → head-to-head comparison
- `generate_interview_questions(candidate_id)` → ~5 screening questions

---

## Project structure

```
Agentic Profile Matching/
├── matching_agent.py          # LangGraph StateGraph + AgentState + nodes
├── app.py                     # Gradio chat interface
├── tools/
│   ├── __init__.py
│   ├── recruitment_tools.py   # Agent-facing tools/adapters (the new work)
│   ├── fs_tools.py            # Milestone 1 file-system tools (reused)
│   ├── resume_rag.py          # Milestone 2 ingestion + ChromaDB (reused)
│   └── job_matcher.py         # Milestone 2 hybrid retrieval + scoring (reused)
├── scripts/
│   └── generate_dataset.py    # Creates synthetic sample resumes (PDF)
├── tests/
│   └── test_scenarios.py      # 6 end-to-end demo flows
├── data/
│   ├── resumes/               # Sample resumes (PDF)
│   └── job_descriptions/      # Sample job descriptions (TXT)
├── requirements.txt
├── .env.example
└── README.md
```

The generated Chroma vector database (`data/chroma_db/`) is **not** committed —
it is built automatically from the resumes in `data/resumes/` on first run.

---

## Setup

### 1. Clone and create a virtual environment

```powershell
git clone <your-repo-url>
cd "Agentic Profile Matching"

python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate         # macOS / Linux
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

> The first install downloads `sentence-transformers` / `torch`, which are
> large — this step can take several minutes.

### 3. Configure your OpenRouter API key

Copy the example env file and add your key:

```powershell
copy .env.example .env               # Windows
# cp .env.example .env               # macOS / Linux
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-or-...your-openrouter-key...
OPENAI_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
```

Get a key from https://openrouter.ai/keys. The app uses the OpenAI-compatible
client pointed at OpenRouter. If no key is set, the agent still runs and falls
back to deterministic, template-based reasoning (no LLM calls).

### 4. Generate sample resumes (if `data/resumes/` is empty)

The repo ships with synthetic sample resumes. To regenerate or add more:

```powershell
python scripts/generate_dataset.py
```

### 5. Build the vector database

This happens **automatically** on the first search. To build it explicitly:

```powershell
python -c "from tools import resume_rag; resume_rag.build_vector_db()"
```

---

## Running the app

```powershell
python app.py
```

Open the printed local URL (default http://127.0.0.1:7860) and chat with the
assistant.

### Demo flow (≈5–6 minutes)

1. *"Find me candidates with React and at least 3 years experience."*
2. *"Compare the top 3."*
3. *"Why did the top candidate rank higher than the second candidate?"*
4. *"Make TypeScript a must-have."*
5. *"Generate interview questions for the top candidate."*

---

## Running the test scenarios

Six end-to-end flows run against the agent in a single shared conversation:

```powershell
python tests/test_scenarios.py
```

Covered flows:

1. Find candidates with React and 3+ years experience.
2. Find candidates with Python, FastAPI, and 3+ years experience.
3. Compare the top 3 candidates.
4. Why did the top candidate rank higher than the second candidate?
5. Make TypeScript a must-have and rerank.
6. Generate interview questions for the top candidate.

---

## Notes

- **Self-contained:** all Milestone 1 / Milestone 2 code needed to run is
  included under `tools/`. The original milestone projects are not required.
- **No secrets committed:** `.env` and the generated `data/chroma_db/` are
  git-ignored. Only `.env.example` is committed.
- **Synthetic data only:** the bundled resumes are fictional and contain no
  real personal information.
- **Portable paths:** the code uses only project-relative paths, so it runs
  the same after a fresh `git clone` on any machine.

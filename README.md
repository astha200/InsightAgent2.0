# InsightAgent

A multi-agent AI system that autonomously analyzes structured datasets, detects anomalies, and generates natural-language summaries with visual reports — using RAG for context-enhanced insights and human-in-the-loop validation.

Built with **LangGraph**, **Claude API**, **Chroma**, **LangSmith**, and **Streamlit**.

---

## Architecture

```
            ┌──────────────────────────────────────────────────────┐
            │              LangGraph Orchestrator                  │
            │           (shared typed state + checkpointer)        │
            └──────────────────────────────────────────────────────┘
                  │              │                │
                  ▼              ▼                ▼
            ┌──────────┐   ┌──────────┐    ┌──────────────┐
            │ Analyst  │──▶│ Reporter │───▶│  Validator   │
            ├──────────┤   ├──────────┤    ├──────────────┤
            │ profile  │   │ chart    │    │ RAG over     │
            │ schema   │   │ select   │    │ domain KB +  │
            │ + ground │   │ + render │    │ user corpus  │
            │ + detect │   │ + caption│    │ + inline     │
            │ + write  │   │ + report │    │ citations    │
            └──────────┘   └──────────┘    └──────────────┘
                  │                                │
                  ▼                                ▼
       [HITL: schema review]              [HITL: insight review]
                  │                                │
                  └────────────┬───────────────────┘
                               ▼
                       ┌──────────────────┐
                       │  Final Report    │
                       └──────────────────┘
                               │
                               ▼
                  ┌────────────────────────────┐
                  │  LLM-as-Judge Evaluation   │
                  │  (5 dimensions, LangSmith) │
                  └────────────────────────────┘
```

The pipeline is a deterministic DAG with two human checkpoints. Each agent has a specific tool surface — pandas/sklearn for the Analyst, matplotlib for the Reporter, vector retrieval for the Validator — and they coordinate through a shared typed state managed by LangGraph.

---

## Features

- **Three specialized agents** with clear contracts and independently testable
- **Domain-grounded RAG** with two collections — canonical knowledge base + user-supplied project context
- **Hybrid anomaly detection** — z-score, IsolationForest, and domain-threshold rules
- **Two human-in-the-loop checkpoints** — schema confirmation and final insight review
- **LLM-as-Judge evaluation** scoring 5 dimensions: retrieval quality, relevance, accuracy, correctness, groundedness
- **LangSmith tracing** end-to-end — every LLM call captured with inputs, outputs, latency, and token counts
- **Prompt caching** on system prompts to reduce per-run cost
- **Mixed-model routing** — Opus for high-stakes reasoning, Sonnet for grounded enrichment, Haiku for cheap captions
- **CLI and Streamlit interfaces** — same orchestrator backs both

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (state machine + `interrupt()` for HITL) |
| LLM | Claude API (Opus, Sonnet, Haiku) |
| Vector store | Chroma (two collections: `domain_kb`, `user_corpus`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim) |
| Anomaly detection | pandas, scipy, scikit-learn (IsolationForest, z-score) |
| Visualization | matplotlib |
| UI | Streamlit |
| Observability + eval | LangSmith (`@traceable` decorator) |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/insightagent.git
cd insightagent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Optional but recommended — enables LangSmith tracing
export LANGSMITH_API_KEY=ls__...
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT=insightagent
```

### 3. Build the vector store (one-time)

```bash
python -m rag.ingest
```

### 4. Run the pipeline

**Streamlit UI:**
```bash
streamlit run app.py
```

**Or CLI:**
```bash
python main.py data/patient_vitals.csv
```

### 5. Run the evaluator

```bash
python -m eval.run_eval
```

---

## Project Structure

```
insightagent/
├── agents/
│   ├── analyst.py        Agent 1 — profile, ground, detect anomalies, synthesize insights
│   ├── reporter.py       Agent 2 — chart selection, rendering, captioning, report assembly
│   └── validator.py      Agent 3 — RAG-grounded enrichment with inline citations
├── rag/
│   ├── ingest.py         Build Chroma collections from domain KB + user corpus
│   ├── retriever.py      Query helper with metadata filtering
│   └── domain_kb/
│       ├── healthcare.md, healthcare.yaml
│       └── finance.md, finance.yaml
├── eval/
│   ├── judge.py          LLM-as-Judge: 5-dimension scoring
│   └── run_eval.py       Eval runner + scored summary table
├── corpus/               User-supplied project context (Markdown)
├── data/                 Sample dataset (synthetic patient vitals)
├── outputs/              Generated artifacts (gitignored)
├── orchestrator.py       LangGraph DAG with 2 HITL interrupts
├── app.py                Streamlit UI
├── main.py               CLI entrypoint
├── state.py              Shared dataclasses (Insight, Finding, ColumnMapping, Citation)
├── llm.py                Claude API wrapper with caching + LangSmith tracing
├── data_gen.py           Synthetic dataset generator
└── requirements.txt
```

---

## How It Works

### Agent 1 — Analyst

1. **Profile** the dataset with pandas (dtypes, null %, cardinality, summary stats).
2. **Ground** each column in the domain knowledge base via RAG — so a column called `bmi` is mapped to "Body Mass Index" with the WHO normal range, not treated as a generic number.
3. **Detect anomalies** — z-score for univariate outliers, IsolationForest for multivariate, YAML thresholds for clinical/domain rules. Severity blended from statistical magnitude + domain bucket.
4. **Synthesize insights** — Opus turns raw findings into structured narrative insights with severity, columns, and finding references.

### Agent 2 — Reporter

1. **Choose chart type** — rule-based on finding type (anomaly → histogram with reference band, correlation → scatter, multivariate → scatter matrix).
2. **Render** to PNG with matplotlib, embedding domain reference ranges.
3. **Caption** each chart with Haiku (cheap and fast for high-volume captions).
4. **Assemble** a Markdown report with charts inline.

### Agent 3 — Validator

1. **Retrieve** top-k passages from two Chroma collections: `domain_kb` (canonical) and `user_corpus` (project-specific notes).
2. **Enrich** each insight with retrieved context — Sonnet weaves the passages into the body with inline `[n]` citation markers.
3. **Cite** — every claim is traceable to a source. If a passage contradicts the original framing, the Validator reframes the insight.

---

## Human-in-the-Loop Checkpoints

Two strategically placed interrupts, not blanket review at every step:

| Checkpoint | What happens | Why here |
|---|---|---|
| **After schema enrichment** | User confirms or edits column mappings | Column misinterpretation cascades — fix it before anomaly detection runs |
| **After Validator** | User accepts, rejects, or edits each enriched insight | Final substantive output review with all context available |

LangGraph's `interrupt()` primitive pauses the graph at these checkpoints; the UI reads the state, the user submits a decision, and `Command(resume=...)` continues from the exact pause point with full state preserved.

---

## Evaluation — LLM-as-Judge

A separate judge agent (Opus) scores each generated insight on five dimensions, each 1–5:

| Dimension | What it measures |
|---|---|
| `retrieval_quality` | Are the retrieved passages genuinely relevant? |
| `relevance` | Does the insight address the underlying finding meaningfully? |
| `accuracy` | Are factual claims consistent with the evidence? |
| `correctness` | Is the domain interpretation right? |
| `groundedness` | Is every claim traceable to a citation? |

Results are written to `outputs/eval_results.json` and printed as a scored summary table. Every judge call is traced to LangSmith with full inputs and outputs, giving a systematic feedback loop for tuning chunking and prompts.

Example output:

```
Insight                              Ret  Rel  Acc  Cor  Gnd   Avg
-----------------------------------------------------
12 patients with morbid obesity...     5    5    4    5    4   4.6
Uncontrolled diabetes detected...      3    4    4    4    4   3.8
-----------------------------------------------------
System average: 4.1 / 5.0
Lowest dimension: retrieval_quality
```

---

## Design Decisions

**Why LangGraph?** The workflow is genuinely a state machine with strict ordering and pause/resume semantics, not autonomous tool-use. LangGraph's typed state and `interrupt()` primitive are the right fit; agent frameworks like LangChain Agents or CrewAI would be overkill.

**Why deterministic detection + LLM interpretation?** A z-score is faster, cheaper, and more reliable than asking an LLM whether a row is unusual. The LLM's value is in *interpretation* — mapping `bmi` to Body Mass Index, framing severity in clinical terms, weaving in retrieved context. The boundary between deterministic code and LLM judgment is the core design choice.

**Why two RAG collections?** Canonical domain knowledge (e.g. WHO BMI ranges) and project-specific notes (e.g. "our cohort runs high BMI because of a bariatric program") serve different roles. Keeping them in separate Chroma collections preserves precedence — stale project notes can't outrank canonical definitions.

**Why hybrid YAML + Markdown for the domain KB?** Numerical thresholds want determinism (a lookup, not a vector query). The Markdown gets embedded for narrative grounding; the YAML drives anomaly severity directly. Clean upgrade path to a metric-definitions table at scale.

**Why heading-based chunking?** The KB documents have explicit structure (one H2 per term). Splitting at H2 boundaries produces semantically complete chunks. Fixed-size chunking would slice mid-definition.

---

## Roadmap

- **Phase 2 — Critic loop**: Validator emits structured critiques (`accept | revise | drop`); orchestrator routes rejected insights back to Analyst or Reporter for revision. Adds genuine multi-agent collaboration without exploding scope.
- **Structured outputs**: replace regex JSON extraction with Anthropic tool-use mode for stricter schema enforcement.
- **Production retrieval**: swap Chroma for pgvector or Qdrant; swap MemorySaver for a Postgres-backed LangGraph checkpointer.
- **Expanded eval set**: curated set of (dataset, expected_findings) pairs for regression testing across releases.

---

## License

MIT

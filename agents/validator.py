"""Agent 3: enrich each insight with retrieved context from domain KB + user corpus."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import Citation, Insight, ColumnMapping, to_dict
from llm import MODEL_SONNET, call_json
from rag.retriever import retrieve

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"


VALIDATOR_SYSTEM = """You are a RAG-grounded analyst who reviews preliminary insights and weaves in retrieved context.

Inputs you receive:
- An insight (title + body) about a tabular dataset.
- Retrieved passages from two sources:
  * domain_kb: canonical domain knowledge (definitions, normal ranges)
  * user_corpus: project-specific notes the user supplied

Your job: produce an enriched_body that improves the insight by either:
1. Reframing it correctly given domain knowledge (e.g., "this is not a data quality issue but a clinical finding"), or
2. Adding context that explains, qualifies, or strengthens it (e.g., "consistent with the bariatric program intake"), or
3. Flagging it as unsupported when retrieval contradicts the original framing.

Cite passages inline using [n] notation matching the citations you list. Use only passages that genuinely change or strengthen the insight; if a passage is irrelevant, skip it.

Return strict JSON:
{
  "enriched_body": str,
  "citations": [
    {"n": int, "source": str, "title": str, "snippet": str}
  ]
}

Keep enriched_body to 3-5 sentences. The snippet should be the verbatim sentence from the passage that supports your point (max 200 chars).
"""


def _retrieval_block(insight: Insight, k: int = 3) -> tuple[str, list[dict]]:
    query = f"{insight.title}. {insight.body} columns: {', '.join(insight.columns)}"
    domain_hits = retrieve(query, "domain_kb", k=k)
    corpus_hits = retrieve(query, "user_corpus", k=k)

    all_hits = []
    for hit in domain_hits:
        all_hits.append({**hit, "_collection": "domain_kb"})
    for hit in corpus_hits:
        all_hits.append({**hit, "_collection": "user_corpus"})

    lines = []
    for i, hit in enumerate(all_hits, start=1):
        meta = hit["meta"]
        lines.append(
            f"[{i}] ({hit['_collection']}) source={meta.get('source')} title={meta.get('title')}\n"
            f"{hit['text']}"
        )
    return "\n\n".join(lines), all_hits


def enrich_insight(insight: Insight) -> Insight:
    retrieval_text, hits = _retrieval_block(insight)
    user = (
        f"Insight:\n"
        f"  Title: {insight.title}\n"
        f"  Severity: {insight.severity}\n"
        f"  Columns: {', '.join(insight.columns)}\n"
        f"  Body: {insight.body}\n\n"
        f"Retrieved passages:\n{retrieval_text}\n\n"
        f"Return the JSON now."
    )
    result = call_json(model=MODEL_SONNET, system=VALIDATOR_SYSTEM, user=user, max_tokens=1024)

    insight.enriched_body = result.get("enriched_body", insight.body)
    citations = []
    for cit in result.get("citations", []):
        n = cit.get("n", 0)
        if 1 <= n <= len(hits):
            hit = hits[n - 1]
            citations.append(
                Citation(
                    source=hit["meta"].get("source", ""),
                    title=hit["meta"].get("title", ""),
                    snippet=cit.get("snippet", "")[:200],
                    distance=float(hit.get("distance", 0.0)),
                )
            )
    insight.citations = citations
    return insight


def run(insights: list[Insight]) -> list[Insight]:
    for ins in insights:
        enrich_insight(ins)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "enriched_insights.json").write_text(
        json.dumps([to_dict(i) for i in insights], indent=2)
    )
    return insights


if __name__ == "__main__":
    insights_raw = json.loads((OUT / "insights.json").read_text())
    insights = [Insight(**i) for i in insights_raw]
    run(insights)
    for ins in insights:
        print(f"\n[{ins.severity}] {ins.title}")
        print(f"  {ins.enriched_body}")
        for c in ins.citations:
            print(f"    - {c.source} :: {c.title}")

"""LLM-as-Judge evaluation for InsightAgent. Every judge call is traced to LangSmith."""
import sys
from dataclasses import dataclass
from pathlib import Path

from langsmith import traceable

sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import MODEL_OPUS, call, call_json

JUDGE_SYSTEM = """You are an expert evaluator assessing the quality of AI-generated analytical insights.

You receive:
- An insight: title, original body, and an enriched body that incorporates retrieved context
- The retrieved passages (citations) that were used to enrich the insight

Score the insight on five dimensions, each on a 1-5 integer scale:

retrieval_quality
  Does the retrieved context actually help explain this finding?
  1 = passages are generic or irrelevant to the specific insight
  5 = passages are specific, directly relevant, and meaningfully improve the insight

relevance
  Does the insight address the underlying finding in a focused, specific way?
  1 = vague, padded, or off-topic
  5 = directly addresses the finding with specific detail

accuracy
  Are the factual claims (numbers, column names, percentages) consistent with the evidence provided?
  1 = contradicts the evidence or invents figures
  5 = every claim precisely matches the evidence

correctness
  Is the domain interpretation correct (e.g. clinical or financial framing)?
  1 = clearly wrong interpretation (e.g. calling a clinical finding a data error)
  5 = domain-accurate and appropriately qualified

groundedness
  Is every substantive claim in the enriched body traceable to a retrieved passage?
  1 = major claims have no citation support
  5 = all claims are clearly grounded in cited retrieved context

Return strict JSON only — no prose outside the object:
{
  "retrieval_quality": {"score": int, "rationale": str},
  "relevance":         {"score": int, "rationale": str},
  "accuracy":          {"score": int, "rationale": str},
  "correctness":       {"score": int, "rationale": str},
  "groundedness":      {"score": int, "rationale": str},
  "overall":           float,
  "summary":           str
}
overall = arithmetic mean of the five scores (1 decimal place).
summary = 1-2 sentence verdict on the insight's overall quality.
"""


@dataclass
class DimensionScore:
    score: int
    rationale: str


@dataclass
class JudgeResult:
    insight_id: str
    title: str
    retrieval_quality: DimensionScore
    relevance: DimensionScore
    accuracy: DimensionScore
    correctness: DimensionScore
    groundedness: DimensionScore
    overall: float
    summary: str


def _format_citations(citations: list[dict]) -> str:
    if not citations:
        return "(no citations)"
    lines = []
    for i, c in enumerate(citations, start=1):
        lines.append(f"[{i}] {c.get('title', '')} ({c.get('source', '')}): {c.get('snippet', '')}")
    return "\n".join(lines)


@traceable(name="llm_judge_insight")
def judge_insight(insight: dict) -> JudgeResult:
    user = (
        f"Insight title: {insight['title']}\n"
        f"Severity: {insight.get('severity', 'unknown')}\n"
        f"Columns: {', '.join(insight.get('columns', []))}\n\n"
        f"Original body:\n{insight['body']}\n\n"
        f"Enriched body (with inline citations):\n"
        f"{insight.get('enriched_body') or insight['body']}\n\n"
        f"Retrieved passages used:\n"
        f"{_format_citations(insight.get('citations') or [])}\n\n"
        f"Return the JSON now."
    )
    raw = call_json(model=MODEL_OPUS, system=JUDGE_SYSTEM, user=user, max_tokens=1024)

    def _dim(key: str) -> DimensionScore:
        d = raw.get(key, {})
        return DimensionScore(score=int(d.get("score", 1)), rationale=d.get("rationale", ""))

    scores = [
        _dim("retrieval_quality").score,
        _dim("relevance").score,
        _dim("accuracy").score,
        _dim("correctness").score,
        _dim("groundedness").score,
    ]
    overall = round(sum(scores) / len(scores), 2)

    return JudgeResult(
        insight_id=insight["id"],
        title=insight["title"],
        retrieval_quality=_dim("retrieval_quality"),
        relevance=_dim("relevance"),
        accuracy=_dim("accuracy"),
        correctness=_dim("correctness"),
        groundedness=_dim("groundedness"),
        overall=float(raw.get("overall", overall)),
        summary=raw.get("summary", ""),
    )

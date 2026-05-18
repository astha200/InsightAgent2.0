"""LangGraph orchestrator: 3 agents wired together with 2 HITL interrupts."""
import json
from pathlib import Path
from typing import TypedDict, Any

import pandas as pd
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from state import ColumnMapping, Finding, Insight, Citation, to_dict
from agents import analyst, reporter, validator
from rag.ingest import load_yaml_kb

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"


class State(TypedDict, total=False):
    csv_path: str
    domain: str
    domain_confidence: float
    mappings: dict[str, dict]
    findings: list[dict]
    insights: list[dict]
    schema_decision: dict
    insight_decisions: list[dict]


def _profile_and_enrich(state: State) -> dict:
    df = pd.read_csv(state["csv_path"])
    profile = analyst.profile_schema(df)
    kb_yaml = load_yaml_kb()
    domain, conf, mappings = analyst.enrich_schema(profile, kb_yaml)
    return {
        "domain": domain,
        "domain_confidence": conf,
        "mappings": {k: to_dict(v) for k, v in mappings.items()},
    }


def _schema_review(state: State) -> dict:
    decision = interrupt(
        {
            "step": "schema_review",
            "domain": state["domain"],
            "domain_confidence": state["domain_confidence"],
            "mappings": state["mappings"],
        }
    )
    return {"schema_decision": decision}


def _apply_schema_decision(state: State) -> dict:
    decision = state.get("schema_decision") or {}
    mappings = {k: dict(v) for k, v in state["mappings"].items()}
    for col, edits in (decision.get("edits") or {}).items():
        if col in mappings:
            mappings[col].update(edits)
    return {
        "mappings": mappings,
        "domain": decision.get("domain", state["domain"]),
    }


def _detect_and_synthesize(state: State) -> dict:
    df = pd.read_csv(state["csv_path"])
    mappings = {k: ColumnMapping(**v) for k, v in state["mappings"].items()}
    findings = analyst.detect_anomalies(df, mappings) + analyst.detect_correlations(df, mappings)
    insights = analyst.synthesize_insights(findings, mappings, state["domain"])
    return {
        "findings": [to_dict(f) for f in findings],
        "insights": [to_dict(i) for i in insights],
    }


def _reporter(state: State) -> dict:
    mappings = {k: ColumnMapping(**v) for k, v in state["mappings"].items()}
    findings = [Finding(**f) for f in state["findings"]]
    insights = [Insight(**i) for i in state["insights"]]
    insights = reporter.run(state["csv_path"], insights, findings, mappings, state["domain"])
    return {"insights": [to_dict(i) for i in insights]}


def _validator(state: State) -> dict:
    insights = [Insight(**_coerce_insight(i)) for i in state["insights"]]
    insights = validator.run(insights)
    return {"insights": [to_dict(i) for i in insights]}


def _coerce_insight(d: dict) -> dict:
    out = dict(d)
    out["citations"] = [
        Citation(**c) if isinstance(c, dict) else c for c in d.get("citations", [])
    ]
    return out


def _insight_review(state: State) -> dict:
    decisions = interrupt(
        {
            "step": "insight_review",
            "insights": state["insights"],
            "domain": state["domain"],
        }
    )
    return {"insight_decisions": decisions}


def _finalize(state: State) -> dict:
    decisions_by_id = {d["id"]: d for d in (state.get("insight_decisions") or [])}

    final_dicts: list[dict] = []
    for ins_dict in state["insights"]:
        d = decisions_by_id.get(ins_dict["id"], {"verdict": "accept"})
        if d.get("verdict") == "reject":
            continue
        if d.get("edited_body"):
            ins_dict = {**ins_dict, "enriched_body": d["edited_body"]}
        final_dicts.append(ins_dict)

    OUT.mkdir(parents=True, exist_ok=True)

    lines = [
        "# InsightAgent Final Report",
        "",
        f"_Domain: **{state['domain']}**_  ·  _Insights accepted: {len(final_dicts)}_",
        "",
        "---",
        "",
    ]
    for ins in final_dicts:
        lines.append(f"## {ins['title']}")
        lines.append(
            f"_Severity: {ins['severity']}_  ·  _Columns: {', '.join(ins['columns'])}_"
        )
        lines.append("")
        lines.append(ins.get("enriched_body") or ins["body"])
        lines.append("")
        if ins.get("chart_path"):
            lines.append(f"![{ins['title']}]({ins['chart_path']})")
            if ins.get("chart_caption"):
                lines.append(f"_{ins['chart_caption']}_")
            lines.append("")
        if ins.get("citations"):
            lines.append("**Sources:**")
            for j, c in enumerate(ins["citations"], start=1):
                lines.append(f"  {j}. _{c['title']}_ ({c['source']}): {c['snippet']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    (OUT / "final_report.md").write_text("\n".join(lines))
    (OUT / "final_insights.json").write_text(json.dumps(final_dicts, indent=2))
    return {}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("profile_enrich", _profile_and_enrich)
    builder.add_node("schema_review", _schema_review)
    builder.add_node("apply_schema", _apply_schema_decision)
    builder.add_node("detect_synthesize", _detect_and_synthesize)
    builder.add_node("reporter", _reporter)
    builder.add_node("validator", _validator)
    builder.add_node("insight_review", _insight_review)
    builder.add_node("finalize", _finalize)

    builder.add_edge(START, "profile_enrich")
    builder.add_edge("profile_enrich", "schema_review")
    builder.add_edge("schema_review", "apply_schema")
    builder.add_edge("apply_schema", "detect_synthesize")
    builder.add_edge("detect_synthesize", "reporter")
    builder.add_edge("reporter", "validator")
    builder.add_edge("validator", "insight_review")
    builder.add_edge("insight_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _status(graph, config) -> dict:
    snap = graph.get_state(config)
    for task in snap.tasks:
        if task.interrupts:
            return {
                "type": "interrupt",
                "payload": task.interrupts[0].value,
                "state": dict(snap.values),
            }
    return {"type": "complete", "state": dict(snap.values)}


def start(csv_path: str, thread_id: str = "default") -> dict:
    g = get_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    g.invoke({"csv_path": csv_path}, config)
    return _status(g, config)


def resume(decision: Any, thread_id: str = "default") -> dict:
    g = get_graph()
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    g.invoke(Command(resume=decision), config)
    return _status(g, config)

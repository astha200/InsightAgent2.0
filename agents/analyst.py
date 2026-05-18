"""Agent 1: profile schema, ground columns in domain KB, detect anomalies, synthesize insights."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import ColumnMapping, Finding, Insight, to_dict
from llm import MODEL_OPUS, call, call_json
from rag.ingest import load_yaml_kb
from rag.retriever import retrieve

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"


def profile_schema(df: pd.DataFrame) -> dict:
    out = {"n_rows": len(df), "columns": {}}
    for col in df.columns:
        s = df[col]
        info: dict = {
            "dtype": str(s.dtype),
            "null_pct": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s):
            non_null = s.dropna()
            info.update(
                {
                    "min": float(non_null.min()) if len(non_null) else None,
                    "max": float(non_null.max()) if len(non_null) else None,
                    "mean": float(non_null.mean()) if len(non_null) else None,
                    "std": float(non_null.std()) if len(non_null) else None,
                    "sample_values": [float(x) for x in non_null.head(5).tolist()],
                }
            )
        else:
            info["sample_values"] = [str(x) for x in s.dropna().head(5).tolist()]
        out["columns"][col] = info
    return out


def _retrieve_domain_context(profile: dict) -> str:
    queries = list(profile["columns"].keys())
    chunks = []
    seen = set()
    for q in queries:
        for r in retrieve(q, "domain_kb", k=2):
            key = (r["meta"]["source"], r["meta"]["title"])
            if key in seen:
                continue
            seen.add(key)
            chunks.append(f"[{r['meta']['domain']} :: {r['meta']['title']}]\n{r['text']}")
    return "\n\n---\n\n".join(chunks)


SCHEMA_SYSTEM = """You are a data-domain analyst. Given a tabular schema and retrieved domain glossary chunks, infer:
1. The most likely domain for the dataset (e.g., "healthcare", "finance"). If unsure, say "unknown".
2. For each column, the canonical concept it represents, its unit, and an expected normal range when applicable.

Return strict JSON only. No prose outside the JSON. Schema:
{
  "domain": str,
  "domain_confidence": float (0.0-1.0),
  "columns": [
    {
      "column": str,
      "concept": str | null,
      "unit": str | null,
      "expected_range": [number, number] | null,
      "confidence": float (0.0-1.0),
      "rationale": str
    }
  ]
}

Be conservative: if a column name is ambiguous (e.g. "score" could mean anything) set concept to null and confidence < 0.5.
"""


def enrich_schema(profile: dict, kb_yaml: dict) -> tuple[str, float, dict[str, ColumnMapping]]:
    domain_context = _retrieve_domain_context(profile)
    user = (
        f"Dataset profile (n_rows={profile['n_rows']}):\n"
        f"{json.dumps(profile['columns'], indent=2)}\n\n"
        f"Retrieved domain reference:\n{domain_context}\n\n"
        f"Return the JSON now."
    )
    result = call_json(model=MODEL_OPUS, system=SCHEMA_SYSTEM, user=user, max_tokens=2048)

    domain = result.get("domain", "unknown")
    domain_conf = float(result.get("domain_confidence", 0.0))

    mappings: dict[str, ColumnMapping] = {}
    yaml_for_domain = kb_yaml.get(domain, {}).get("columns", {}) if domain in kb_yaml else {}

    for col_info in result.get("columns", []):
        col = col_info["column"]
        if col not in profile["columns"]:
            continue
        m = ColumnMapping(
            column=col,
            concept=col_info.get("concept"),
            unit=col_info.get("unit"),
            expected_range=col_info.get("expected_range"),
            domain=domain if col_info.get("concept") else None,
            confidence=float(col_info.get("confidence", 0.0)),
            rationale=col_info.get("rationale", ""),
        )
        yaml_match = None
        for yaml_col, yaml_info in yaml_for_domain.items():
            if col == yaml_col or col in yaml_info.get("aliases", []):
                yaml_match = yaml_info
                break
        if yaml_match:
            m.expected_range = yaml_match.get("expected_range") or m.expected_range
            m.severity_thresholds = yaml_match.get("severity_thresholds")
            if not m.unit:
                m.unit = yaml_match.get("unit")
            if not m.concept:
                m.concept = yaml_match.get("concept")
        mappings[col] = m

    for col in profile["columns"]:
        if col not in mappings:
            mappings[col] = ColumnMapping(column=col)
    return domain, domain_conf, mappings


def _classify_severity(value: float, thresholds: dict | None) -> str:
    if not thresholds:
        return "low"
    for label in ("extreme", "moderate", "mild"):
        rng = thresholds.get(label)
        if rng and rng[0] <= value <= rng[1]:
            return label if label != "mild" else "low"
    return "low"


def detect_anomalies(df: pd.DataFrame, mappings: dict[str, ColumnMapping]) -> list[Finding]:
    findings: list[Finding] = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        m = mappings.get(col, ColumnMapping(column=col))

        std = s.std() or 1.0
        z = (s - s.mean()) / std
        stat_outliers = s[z.abs() > 3]

        domain_violations = pd.Series(dtype=float)
        worst_severity = "low"
        bucket_counts = {"mild": 0, "moderate": 0, "extreme": 0}
        if m.expected_range:
            lo, hi = m.expected_range
            domain_violations = s[(s < lo) | (s > hi)]

        if m.severity_thresholds:
            for label, rng in m.severity_thresholds.items():
                lo, hi = rng
                bucket_counts[label] = int(((s >= lo) & (s <= hi)).sum())
            for label in ("extreme", "moderate", "mild"):
                if bucket_counts.get(label, 0) > 0:
                    worst_severity = "high" if label == "extreme" else (
                        "moderate" if label == "moderate" else "low"
                    )
                    if label == "extreme":
                        worst_severity = "extreme"
                    break

        if len(stat_outliers) == 0 and len(domain_violations) == 0:
            continue

        findings.append(
            Finding(
                id=f"anom-{col}",
                type="anomaly",
                columns=[col],
                severity=worst_severity if worst_severity != "low" else (
                    "moderate" if len(stat_outliers) >= 5 else "low"
                ),
                evidence={
                    "n_stat_outliers": int(len(stat_outliers)),
                    "n_domain_violations": int(len(domain_violations)),
                    "n_total": int(len(s)),
                    "mean": float(s.mean()),
                    "std": float(s.std()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                    "expected_range": m.expected_range,
                    "severity_buckets": bucket_counts,
                    "extreme_examples": [
                        float(x) for x in stat_outliers.head(5).tolist()
                    ],
                },
            )
        )

    if len(numeric_cols) >= 2:
        sub = df[numeric_cols].dropna()
        if len(sub) >= 20:
            iso = IsolationForest(contamination=0.05, random_state=0).fit(sub)
            scores = iso.decision_function(sub)
            n_multi = int((scores < np.percentile(scores, 5)).sum())
            findings.append(
                Finding(
                    id="anom-multivariate",
                    type="multivariate_anomaly",
                    columns=numeric_cols,
                    severity="moderate",
                    evidence={
                        "n_outliers": n_multi,
                        "n_total": int(len(sub)),
                        "method": "IsolationForest contamination=0.05",
                    },
                )
            )
    return findings


def detect_correlations(df: pd.DataFrame, mappings: dict[str, ColumnMapping]) -> list[Finding]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        return []
    corr = df[numeric_cols].corr().abs()
    findings = []
    seen = set()
    for a in numeric_cols:
        for b in numeric_cols:
            if a >= b:
                continue
            r = float(corr.loc[a, b])
            if r >= 0.5 and (a, b) not in seen:
                seen.add((a, b))
                findings.append(
                    Finding(
                        id=f"corr-{a}-{b}",
                        type="correlation",
                        columns=[a, b],
                        severity="moderate" if r >= 0.7 else "low",
                        evidence={
                            "pearson_r": float(df[[a, b]].corr().iloc[0, 1]),
                            "abs_r": r,
                            "n": int(df[[a, b]].dropna().shape[0]),
                        },
                    )
                )
    return findings


INSIGHT_SYSTEM = """You are a domain-aware data analyst writing insight cards for stakeholders.

For each statistical finding you receive, produce ONE insight with:
- title: a 6-12 word headline
- body: 2-4 sentences. Frame in the dataset's domain (e.g. clinical, financial). Reference numbers from the evidence. Avoid jargon padding.
- severity: one of "low" | "moderate" | "high" | "extreme" — preserve the input severity.
- finding_ids: list with the source finding id.
- columns: the columns involved.

Do NOT invent numbers. Do NOT moralize. Do NOT recommend specific actions; describe what the data shows and why it matters in context.

Return strict JSON only:
{
  "insights": [
    {"id": str, "title": str, "body": str, "severity": str, "finding_ids": [str], "columns": [str]}
  ]
}
Use insight ids of the form "ins-1", "ins-2", ...
"""


def synthesize_insights(
    findings: list[Finding],
    mappings: dict[str, ColumnMapping],
    domain: str,
) -> list[Insight]:
    if not findings:
        return []
    user = (
        f"Domain: {domain}\n\n"
        f"Column mappings:\n{json.dumps({k: to_dict(v) for k, v in mappings.items()}, indent=2)}\n\n"
        f"Findings:\n{json.dumps([to_dict(f) for f in findings], indent=2)}\n\n"
        f"Return the JSON now."
    )
    result = call_json(model=MODEL_OPUS, system=INSIGHT_SYSTEM, user=user, max_tokens=4096)
    insights = []
    for item in result.get("insights", []):
        insights.append(
            Insight(
                id=item["id"],
                title=item["title"],
                body=item["body"],
                severity=item.get("severity", "low"),
                finding_ids=item.get("finding_ids", []),
                columns=item.get("columns", []),
            )
        )
    return insights


def run(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    profile = profile_schema(df)
    kb_yaml = load_yaml_kb()
    domain, domain_conf, mappings = enrich_schema(profile, kb_yaml)

    findings = detect_anomalies(df, mappings) + detect_correlations(df, mappings)
    insights = synthesize_insights(findings, mappings, domain)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "schema.json").write_text(
        json.dumps(
            {
                "domain": domain,
                "domain_confidence": domain_conf,
                "mappings": {k: to_dict(v) for k, v in mappings.items()},
            },
            indent=2,
        )
    )
    (OUT / "findings.json").write_text(
        json.dumps([to_dict(f) for f in findings], indent=2)
    )
    (OUT / "insights.json").write_text(
        json.dumps([to_dict(i) for i in insights], indent=2)
    )
    return {
        "domain": domain,
        "domain_confidence": domain_conf,
        "mappings": mappings,
        "findings": findings,
        "insights": insights,
    }


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/patient_vitals.csv"
    result = run(csv)
    print(f"Domain: {result['domain']} (conf={result['domain_confidence']:.2f})")
    print(f"Findings: {len(result['findings'])}")
    print(f"Insights: {len(result['insights'])}")
    for ins in result["insights"]:
        print(f"  [{ins.severity}] {ins.title}")

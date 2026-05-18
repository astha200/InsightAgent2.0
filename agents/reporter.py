"""Agent 2: render charts per insight, generate captions, assemble report."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from state import Finding, Insight, ColumnMapping, to_dict
from llm import MODEL_HAIKU, call

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"
CHART_DIR = OUT / "charts"


CAPTION_SYSTEM = """You write concise figure captions for analytical reports.
Given an insight title, body, and a description of the rendered chart, write 1-2 sentences (max 40 words) describing what the figure shows. Plain prose, no markdown, no preface like "This chart shows".
Return only the caption text."""


def _finding_by_id(findings: list[Finding], fid: str) -> Finding | None:
    for f in findings:
        if f.id == fid:
            return f
    return None


def _render_anomaly(df: pd.DataFrame, col: str, mapping: ColumnMapping, out_path: Path) -> str:
    s = df[col].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(s, bins=30, color="#4682B4", alpha=0.85, edgecolor="white")
    if mapping.expected_range:
        lo, hi = mapping.expected_range
        ax.axvspan(lo, hi, color="#9DC183", alpha=0.25, label=f"Expected {lo}-{hi}")
    if mapping.severity_thresholds:
        for label, rng in mapping.severity_thresholds.items():
            if label == "extreme":
                ax.axvspan(rng[0], rng[1], color="#E07A5F", alpha=0.25, label=f"Extreme >= {rng[0]}")
                break
    ax.set_xlabel(f"{col} ({mapping.unit or 'n/a'})")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {mapping.concept or col}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return (
        f"Histogram of {col} (n={len(s)}). "
        f"Mean={s.mean():.2f}, std={s.std():.2f}, range=[{s.min():.2f}, {s.max():.2f}]. "
        + (f"Reference range {mapping.expected_range}." if mapping.expected_range else "")
    )


def _render_correlation(df: pd.DataFrame, cols: list[str], out_path: Path) -> str:
    a, b = cols[0], cols[1]
    sub = df[[a, b]].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub[a], sub[b], s=14, alpha=0.55, color="#4682B4")
    r = sub.corr().iloc[0, 1]
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{a} vs {b} (r = {r:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return f"Scatter plot of {a} vs {b} on {len(sub)} rows. Pearson r = {r:.2f}."


def _render_multivariate(df: pd.DataFrame, cols: list[str], out_path: Path) -> str:
    cols = cols[:4]
    sub = df[cols].dropna()
    fig, ax = plt.subplots(figsize=(7, 5))
    pd.plotting.scatter_matrix(sub, ax=ax, alpha=0.4, diagonal="hist")
    fig.suptitle("Pairwise relationships (multivariate scan)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return f"Scatter matrix over {cols} on {len(sub)} rows."


def render_chart(
    insight: Insight,
    findings: list[Finding],
    df: pd.DataFrame,
    mappings: dict[str, ColumnMapping],
) -> tuple[str, str]:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fid = insight.finding_ids[0] if insight.finding_ids else None
    f = _finding_by_id(findings, fid) if fid else None
    out_path = CHART_DIR / f"{insight.id}.png"

    if f is None or f.type == "anomaly":
        col = insight.columns[0] if insight.columns else (f.columns[0] if f else None)
        if col is None or col not in df.columns:
            return "", ""
        desc = _render_anomaly(df, col, mappings.get(col, ColumnMapping(column=col)), out_path)
    elif f.type == "correlation" and len(f.columns) >= 2:
        desc = _render_correlation(df, f.columns, out_path)
    elif f.type == "multivariate_anomaly":
        desc = _render_multivariate(df, f.columns, out_path)
    else:
        return "", ""

    return str(out_path.relative_to(ROOT)), desc


def generate_caption(insight: Insight, chart_desc: str) -> str:
    user = (
        f"Insight title: {insight.title}\n"
        f"Insight body: {insight.body}\n"
        f"Chart: {chart_desc}\n"
    )
    return call(model=MODEL_HAIKU, system=CAPTION_SYSTEM, user=user, max_tokens=120).strip()


def assemble_report(insights: list[Insight], domain: str) -> str:
    lines = [
        f"# InsightAgent Report",
        f"",
        f"_Domain detected: **{domain}**_",
        f"",
        f"---",
        f"",
    ]
    for ins in insights:
        lines.append(f"## {ins.title}")
        lines.append(f"_Severity: {ins.severity}_  ·  _Columns: {', '.join(ins.columns)}_")
        lines.append("")
        lines.append(ins.body)
        lines.append("")
        if ins.chart_path:
            lines.append(f"![{ins.title}]({ins.chart_path})")
            lines.append("")
            if ins.chart_caption:
                lines.append(f"_{ins.chart_caption}_")
                lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)


def run(
    csv_path: str,
    insights: list[Insight],
    findings: list[Finding],
    mappings: dict[str, ColumnMapping],
    domain: str,
) -> list[Insight]:
    df = pd.read_csv(csv_path)
    for ins in insights:
        path, desc = render_chart(ins, findings, df, mappings)
        if path:
            ins.chart_path = path
            ins.chart_caption = generate_caption(ins, desc)

    OUT.mkdir(parents=True, exist_ok=True)
    report_md = assemble_report(insights, domain)
    (OUT / "report.md").write_text(report_md)
    (OUT / "insights.json").write_text(
        json.dumps([to_dict(i) for i in insights], indent=2)
    )
    return insights


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/patient_vitals.csv"
    insights_raw = json.loads((OUT / "insights.json").read_text())
    findings_raw = json.loads((OUT / "findings.json").read_text())
    schema = json.loads((OUT / "schema.json").read_text())
    insights = [Insight(**i) for i in insights_raw]
    findings = [Finding(**f) for f in findings_raw]
    mappings = {
        k: ColumnMapping(**v) for k, v in schema["mappings"].items()
    }
    run(csv, insights, findings, mappings, schema["domain"])
    print(f"Wrote report to {OUT / 'report.md'}")
    print(f"Charts in {CHART_DIR}")

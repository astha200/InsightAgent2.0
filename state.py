from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class ColumnMapping:
    column: str
    concept: str | None = None
    unit: str | None = None
    expected_range: list[float] | None = None
    severity_thresholds: dict[str, list[float]] | None = None
    domain: str | None = None
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class Finding:
    id: str
    type: str
    columns: list[str]
    severity: str
    evidence: dict[str, Any]


@dataclass
class Citation:
    source: str
    title: str
    snippet: str
    distance: float


@dataclass
class Insight:
    id: str
    title: str
    body: str
    severity: str
    finding_ids: list[str]
    columns: list[str]
    chart_path: str | None = None
    chart_caption: str | None = None
    enriched_body: str | None = None
    citations: list[Citation] = field(default_factory=list)


def to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj

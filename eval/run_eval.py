"""Run LLM-as-Judge over enriched insights and write scored results to outputs/eval_results.json."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval.judge import JudgeResult, judge_insight

ROOT = Path(__file__).parent.parent
OUT = ROOT / "outputs"

DIMS = ["retrieval_quality", "relevance", "accuracy", "correctness", "groundedness"]
DIM_LABELS = {"retrieval_quality": "Ret", "relevance": "Rel", "accuracy": "Acc",
              "correctness": "Cor", "groundedness": "Gnd"}


def _result_to_dict(r: JudgeResult) -> dict:
    d = asdict(r)
    return d


def _print_table(results: list[dict]) -> None:
    header = f"{'Insight':<36}" + "".join(f"{DIM_LABELS[d]:>5}" for d in DIMS) + f"{'Avg':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['title'][:35]:<36}"
        for d in DIMS:
            row += f"{r[d]['score']:>5}"
        row += f"{r['overall']:>6.1f}"
        print(row)
    print("=" * len(header))
    if results:
        avg = sum(r["overall"] for r in results) / len(results)
        print(f"\nSystem average: {avg:.2f} / 5.0")
        worst_dim = min(DIMS, key=lambda d: sum(r[d]["score"] for r in results))
        print(f"Lowest dimension: {worst_dim} — tune chunking or prompts here first")


def run_eval(insights_path: str | None = None) -> list[dict]:
    path = Path(insights_path) if insights_path else OUT / "enriched_insights.json"
    if not path.exists():
        print(f"No insights found at {path}. Run the pipeline first.")
        return []

    insights = json.loads(path.read_text())
    if not insights:
        print("No insights to evaluate.")
        return []

    print(f"Evaluating {len(insights)} insights with LLM-as-Judge...")
    print("(Runs are traced to LangSmith if LANGCHAIN_TRACING_V2=true)\n")

    results = []
    for ins in insights:
        print(f"  Judging: {ins['title'][:55]}...")
        r = judge_insight(ins)
        d = _result_to_dict(r)
        results.append(d)
        print(f"    Overall {r.overall:.1f}/5 — {r.summary}")

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nWrote scores to {out_path}")

    _print_table(results)
    return results


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_eval(path)

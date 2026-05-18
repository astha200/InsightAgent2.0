"""CLI entrypoint: runs the full pipeline with text-based HITL."""
import sys
from pathlib import Path

import orchestrator

ROOT = Path(__file__).parent


def _print_schema_review(payload: dict) -> dict:
    print("\n" + "=" * 60)
    print("SCHEMA REVIEW")
    print("=" * 60)
    print(f"Detected domain: {payload['domain']} (confidence {payload['domain_confidence']:.2f})\n")
    print(f"{'Column':<18}{'Concept':<28}{'Unit':<12}{'Conf':<6}")
    print("-" * 64)
    for col, m in payload["mappings"].items():
        print(
            f"{col:<18}{(m.get('concept') or '(unmapped)'):<28}"
            f"{(m.get('unit') or '-'):<12}{m.get('confidence', 0.0):<6.2f}"
        )
    input("\nPress Enter to confirm and continue (CLI mode does not support edits)...")
    return {"approved": True, "edits": {}, "domain": payload["domain"]}


def _print_insight_review(payload: dict) -> list[dict]:
    print("\n" + "=" * 60)
    print(f"INSIGHT REVIEW ({len(payload['insights'])} insights)")
    print("=" * 60)
    decisions = []
    for ins in payload["insights"]:
        print(f"\n[{ins['severity'].upper()}] {ins['title']}")
        print(f"  Columns: {', '.join(ins['columns'])}")
        print(f"  {ins.get('enriched_body') or ins['body']}")
        for c in ins.get("citations") or []:
            print(f"    - {c['title']} ({c['source']}): {c['snippet']}")
        ans = input("  [a]ccept (default) / [r]eject > ").strip().lower() or "a"
        verdict = "reject" if ans.startswith("r") else "accept"
        decisions.append({"id": ins["id"], "verdict": verdict, "edited_body": None})
    return decisions


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else "data/patient_vitals.csv"
    thread = "cli"
    print(f"Running pipeline on {csv}...")
    result = orchestrator.start(csv, thread)

    while result["type"] == "interrupt":
        step = result["payload"]["step"]
        if step == "schema_review":
            decision = _print_schema_review(result["payload"])
        elif step == "insight_review":
            decision = _print_insight_review(result["payload"])
        else:
            raise RuntimeError(f"Unknown interrupt step: {step}")
        result = orchestrator.resume(decision, thread)

    out = ROOT / "outputs" / "final_report.md"
    print(f"\nDone. Final report: {out}")


if __name__ == "__main__":
    main()

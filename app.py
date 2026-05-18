"""Streamlit UI for InsightAgent. Run with: streamlit run app.py"""
import uuid
from pathlib import Path

import streamlit as st

import orchestrator

ROOT = Path(__file__).parent

st.set_page_config(page_title="InsightAgent", layout="wide")
st.title("InsightAgent")
st.caption("Multi-agent insight generator with domain-grounded RAG and human-in-the-loop validation")


def _init_session():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.stage = "init"
        st.session_state.payload = None


def _reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    _init_session()


def _advance(decision):
    result = orchestrator.resume(decision, st.session_state.thread_id)
    if result["type"] == "interrupt":
        st.session_state.stage = result["payload"]["step"]
        st.session_state.payload = result["payload"]
    else:
        st.session_state.stage = "complete"
        st.session_state.payload = None


_init_session()

with st.sidebar:
    st.header("Configuration")
    csv_path = st.text_input("Dataset CSV path", value="data/patient_vitals.csv")
    st.divider()
    stage = st.session_state.stage
    progress_map = {
        "init": (0, "Ready"),
        "schema_review": (1, "Awaiting schema confirmation"),
        "insight_review": (2, "Awaiting insight review"),
        "complete": (3, "Complete"),
    }
    step_idx, step_label = progress_map.get(stage, (0, stage))
    st.progress(step_idx / 3, text=step_label)
    if st.button("Reset session"):
        _reset()
        st.rerun()


if st.session_state.stage == "init":
    st.subheader("Run a new analysis")
    st.write(
        "Click below to launch the pipeline. "
        "The Analyst agent will profile the dataset and ground each column in the domain knowledge base, "
        "then pause for your confirmation."
    )
    if st.button("Run analysis", type="primary"):
        with st.spinner("Profiling dataset and grounding schema in domain knowledge..."):
            result = orchestrator.start(csv_path, st.session_state.thread_id)
        if result["type"] == "interrupt":
            st.session_state.stage = result["payload"]["step"]
            st.session_state.payload = result["payload"]
        else:
            st.session_state.stage = "complete"
        st.rerun()


elif st.session_state.stage == "schema_review":
    payload = st.session_state.payload
    st.subheader("Checkpoint 1: Schema review")
    st.markdown(
        f"**Detected domain:** `{payload['domain']}`  "
        f"·  **Confidence:** `{payload['domain_confidence']:.2f}`"
    )
    st.caption(
        "Review the column mappings inferred by the Analyst. "
        "Edit any concept or unit you want to override before anomaly detection runs."
    )

    edits: dict[str, dict] = {}
    domain_override = st.text_input("Override domain (optional)", value=payload["domain"])

    for col, m in payload["mappings"].items():
        with st.expander(f"`{col}` → {m.get('concept') or '(unmapped)'}", expanded=False):
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                new_concept = st.text_input(
                    "Concept", value=m.get("concept") or "", key=f"concept_{col}"
                )
            with c2:
                new_unit = st.text_input(
                    "Unit", value=m.get("unit") or "", key=f"unit_{col}"
                )
            with c3:
                st.metric("LLM confidence", f"{m.get('confidence', 0.0):.2f}")
            if m.get("rationale"):
                st.caption(f"_Rationale: {m['rationale']}_")
            if m.get("expected_range"):
                st.caption(f"Expected range: {m['expected_range']}")

            new_concept_norm = new_concept or None
            new_unit_norm = new_unit or None
            if new_concept_norm != m.get("concept"):
                edits.setdefault(col, {})["concept"] = new_concept_norm
            if new_unit_norm != m.get("unit"):
                edits.setdefault(col, {})["unit"] = new_unit_norm

    if st.button("Confirm and continue", type="primary"):
        with st.spinner("Detecting anomalies, rendering charts, retrieving context..."):
            _advance({"approved": True, "edits": edits, "domain": domain_override})
        st.rerun()


elif st.session_state.stage == "insight_review":
    payload = st.session_state.payload
    st.subheader("Checkpoint 2: Insight review")
    st.caption(
        f"{len(payload['insights'])} insights generated. "
        "Each one has been enriched with retrieved context. Accept, reject, or edit each."
    )

    severity_color = {
        "extreme": "red",
        "high": "red",
        "moderate": "orange",
        "low": "green",
    }

    decisions: list[dict] = []
    for ins in payload["insights"]:
        with st.container(border=True):
            top = st.columns([4, 1])
            with top[0]:
                badge = severity_color.get(ins["severity"], "gray")
                st.markdown(f"### {ins['title']}")
                st.markdown(
                    f":{badge}[**{ins['severity'].upper()}**]  "
                    f"·  Columns: `{', '.join(ins['columns'])}`"
                )
                st.write(ins.get("enriched_body") or ins["body"])
                if ins.get("chart_path"):
                    full = ROOT / ins["chart_path"]
                    if full.exists():
                        st.image(str(full), caption=ins.get("chart_caption") or "")
                if ins.get("citations"):
                    with st.expander("Sources retrieved"):
                        for c in ins["citations"]:
                            st.markdown(
                                f"- **{c['title']}** _(`{c['source']}`)_: {c['snippet']}"
                            )
            with top[1]:
                verdict = st.radio(
                    "Decision",
                    options=["accept", "reject"],
                    index=0,
                    key=f"verdict_{ins['id']}",
                )
                edited = st.text_area(
                    "Edit body (optional)",
                    value="",
                    key=f"edit_{ins['id']}",
                    height=110,
                    placeholder="Leave blank to keep current body",
                )
            decisions.append(
                {
                    "id": ins["id"],
                    "verdict": verdict,
                    "edited_body": edited or None,
                }
            )

    if st.button("Submit all decisions", type="primary"):
        with st.spinner("Finalizing report..."):
            _advance(decisions)
        st.rerun()


elif st.session_state.stage == "complete":
    st.subheader("Final report")
    final = ROOT / "outputs" / "final_report.md"
    if final.exists():
        st.markdown(final.read_text(), unsafe_allow_html=False)
        st.divider()
        st.download_button(
            "Download final_report.md",
            data=final.read_text(),
            file_name="final_report.md",
        )
    else:
        st.info("No report was generated.")

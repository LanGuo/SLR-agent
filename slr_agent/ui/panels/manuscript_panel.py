# slr_agent/ui/panels/manuscript_panel.py
import gradio as gr


def build_manuscript_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 7 manuscript checkpoint panel with rubric-guided revision loop.

    data keys:
      draft (str) — full manuscript markdown
      rubric (dict) — {template: {...}, scores: [{criterion, score, explanation}]}
      draft_version (int)
    """
    draft = data.get("draft", "")
    rubric = data.get("rubric", {})
    scores = rubric.get("scores", [])
    template = rubric.get("template", {})
    version = data.get("draft_version", 1)

    with gr.Column() as panel:
        gr.Markdown(f"## Stage 7: Manuscript Review (Draft v{version})")

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("### Manuscript Draft")
                draft_edit = gr.Textbox(
                    label="Draft (editable — changes saved on Approve)",
                    value=draft,
                    lines=30,
                )

            with gr.Column(scale=2):
                gr.Markdown("### Rubric Scores")
                score_displays = []
                for s in scores:
                    color = {"met": "🟢", "partial": "🟡", "not met": "🔴"}.get(s.get("score", ""), "⚪")
                    gr.Markdown(
                        f"{color} **{s.get('criterion', '')}**\n\n"
                        f"*{s.get('explanation', '')}*"
                    )

                gr.Markdown("### Rubric Criteria (editable)")
                all_criteria = [
                    c
                    for sec in template.get("sections", [])
                    for c in sec.get("rubric_criteria", [])
                ]
                rubric_edit = gr.Textbox(
                    label="One criterion per line",
                    value="\n".join(all_criteria),
                    lines=10,
                )

                gr.Markdown("### Paste journal guidelines (optional)")
                extra_rubric = gr.Textbox(
                    label="Additional rubric / journal submission guidelines",
                    lines=5,
                    placeholder="Paste guidelines here to add to revision criteria...",
                )

        with gr.Row():
            approve_btn = gr.Button("Approve & Finalise", variant="primary", scale=2)
            revise_btn = gr.Button("Trigger LLM Revision", variant="secondary", scale=2)

        status_out = gr.Textbox(label="Status", value="Review draft and rubric scores.", interactive=False)

        _resumed = [False]  # guard against double-resume

        def on_approve(draft_text):
            if _resumed[0]:
                return "Already submitted — waiting for pipeline."
            _resumed[0] = True
            ui_handler.resume({**data, "action": "approve", "draft": draft_text})
            return "Finalised. Generating Word export..."

        def on_revise(rubric_text, extra_text, draft_text):
            if _resumed[0]:
                return "Already submitted — waiting for pipeline."
            _resumed[0] = True
            criteria = [c.strip() for c in rubric_text.split("\n") if c.strip()]
            if extra_text.strip():
                criteria += [c.strip() for c in extra_text.split("\n") if c.strip()]
            updated_template = {
                **template,
                "sections": _redistribute_criteria(template.get("sections", []), criteria),
            }
            ui_handler.resume({
                **data,
                "action": "revise",
                "template": updated_template,
                "draft": draft_text,
            })
            return "Revision requested. Waiting for new draft..."

        approve_btn.click(on_approve, inputs=[draft_edit], outputs=[status_out])
        revise_btn.click(on_revise, inputs=[rubric_edit, extra_rubric, draft_edit], outputs=[status_out])

    return panel


def _redistribute_criteria(sections: list, flat_criteria: list) -> list:
    """Distribute flat criteria list back across sections evenly."""
    if not sections:
        return sections
    per_section = max(1, len(flat_criteria) // len(sections))
    result = []
    for i, sec in enumerate(sections):
        start = i * per_section
        end = start + per_section if i < len(sections) - 1 else len(flat_criteria)
        result.append({**sec, "rubric_criteria": flat_criteria[start:end]})
    return result

# slr_agent/ui/panels/extraction.py
import gradio as gr


def build_extraction_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 5 extraction checkpoint panel.

    data keys: papers (list of {pmid, title, extracted_data, quarantined_fields, grade_score})
    """
    papers = data.get("papers", [])

    with gr.Column() as panel:
        gr.Markdown(f"## Stage 5: Extraction\n**Papers:** {len(papers)}")
        gr.Markdown("Edit extracted fields below. Changes are saved as human overrides.")

        field_inputs = []  # list of (pmid, field_name, gr.Textbox)
        for p in papers:
            with gr.Accordion(f"{p.get('pmid', '')} — {p.get('title', '')[:60]}", open=False):
                grade = p.get("grade_score", {})
                gr.Markdown(f"**GRADE certainty:** {grade.get('certainty', 'N/A')} | "
                            f"Risk of bias: {grade.get('risk_of_bias', 'N/A')}")
                for field_name, value in (p.get("extracted_data") or {}).items():
                    tb = gr.Textbox(label=field_name, value=str(value), lines=2)
                    field_inputs.append((p["pmid"], field_name, tb))
                quarantined = p.get("quarantined_fields", [])
                if quarantined:
                    gr.Markdown(f"**Quarantined fields ({len(quarantined)}):**")
                    for qf in quarantined:
                        gr.Markdown(
                            f"- `{qf.get('field_name')}`: `{qf.get('value')}` "
                            f"— *{qf.get('reason', '')}*"
                        )

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        status_out = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(*values):
            # Rebuild extracted_data per paper from edited textboxes
            edited_papers = []
            value_idx = 0
            for p in papers:
                fields = p.get("extracted_data") or {}
                updated_fields = {}
                for field_name in fields:
                    updated_fields[field_name] = values[value_idx] if value_idx < len(values) else fields[field_name]
                    value_idx += 1
                edited_papers.append({**p, "extracted_data": updated_fields})
            ui_handler.resume({**data, "papers": edited_papers, "action": "approve"})
            return "Approved. Pipeline continuing..."

        tb_inputs = [tb for _, _, tb in field_inputs]
        approve_btn.click(on_approve, inputs=tb_inputs, outputs=[status_out])

    return panel

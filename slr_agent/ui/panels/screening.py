# slr_agent/ui/panels/screening.py
import gradio as gr


def build_screening_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 3 screening checkpoint panel.

    data keys: papers (list of {pmid, title, abstract, decision, reason}),
               n_included, n_excluded, n_uncertain
    """
    papers = data.get("papers", [])
    included = [p for p in papers if p.get("decision") == "include"]
    excluded = [p for p in papers if p.get("decision") == "exclude"]

    with gr.Column() as panel:
        gr.Markdown(
            f"## Stage 3: Screening\n"
            f"**Included:** {len(included)} | **Excluded:** {len(excluded)}"
        )
        gr.Markdown("Flip decisions below. Papers added here are marked **include** directly.")

        decision_dropdowns = []
        with gr.Accordion("Included papers", open=True):
            for p in included[:100]:
                with gr.Row():
                    gr.Markdown(
                        f"**{p.get('pmid', '')}** — {p.get('title', '')[:80]}\n\n"
                        f"*{p.get('abstract', '')[:200]}...*\n\n"
                        f"Reason: {p.get('reason', '')}",
                        scale=4,
                    )
                    dd = gr.Dropdown(
                        choices=["include", "exclude"],
                        value=p.get("decision", "include"),
                        label="Decision",
                        scale=1,
                    )
                    decision_dropdowns.append((p["pmid"], dd))

        with gr.Accordion("Excluded papers", open=False):
            for p in excluded[:100]:
                with gr.Row():
                    gr.Markdown(
                        f"**{p.get('pmid', '')}** — {p.get('title', '')[:80]}\n\n"
                        f"Reason: {p.get('reason', '')}",
                        scale=4,
                    )
                    dd = gr.Dropdown(
                        choices=["include", "exclude"],
                        value=p.get("decision", "exclude"),
                        label="Decision",
                        scale=1,
                    )
                    decision_dropdowns.append((p["pmid"], dd))

        gr.Markdown("### Add a paper manually (marked include directly)")
        with gr.Row():
            pmid_input = gr.Textbox(label="PMID", placeholder="12345678", scale=2)
            add_btn = gr.Button("Add by PMID", scale=1)
        add_status = gr.Textbox(label="", interactive=False)
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        pdf_title = gr.Textbox(label="Title for uploaded PDF")

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        status_out = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(*dropdown_values):
            updated = []
            for i, (pmid, _) in enumerate(decision_dropdowns):
                orig = next((p for p in papers if p["pmid"] == pmid), {})
                updated.append({
                    **orig,
                    "decision": dropdown_values[i] if i < len(dropdown_values) else orig.get("decision"),
                })
            ui_handler.resume({**data, "papers": updated, "action": "approve"})
            return "Approved. Pipeline continuing..."

        def on_add_pmid(pmid_val):
            if not pmid_val.strip():
                return "Enter a PMID first."
            return f"PMID {pmid_val.strip()} queued for manual include."

        add_btn.click(on_add_pmid, inputs=[pmid_input], outputs=[add_status])
        dd_inputs = [dd for _, dd in decision_dropdowns]
        approve_btn.click(on_approve, inputs=dd_inputs, outputs=[status_out])

    return panel

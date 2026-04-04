# slr_agent/ui/panels/search.py
import gradio as gr


def build_search_panel(data: dict, ui_handler, pending) -> tuple:
    """Build the stage 2 search checkpoint panel.

    data keys: papers (list of {pmid, title, source}), n_retrieved, n_duplicates_removed
    Returns (approve_fn, gr.Column) for embedding in the checkpoint area.
    """
    papers = data.get("papers", [])

    with gr.Column() as panel:
        gr.Markdown(f"## Stage 2: Search Results\n"
                    f"Retrieved: **{data.get('n_retrieved', len(papers))}** papers "
                    f"(after removing **{data.get('n_duplicates_removed', 0)}** duplicates)")

        cap_note = f" (showing first 200 of {len(papers)} — remaining papers will proceed to screening)" if len(papers) > 200 else ""
        gr.Markdown(f"Uncheck papers to exclude them before screening. Add missing papers below.{cap_note}")

        # Paper checkboxes
        paper_checks = []
        for p in papers[:200]:  # cap at 200 for UI performance
            cb = gr.Checkbox(
                label=f"[{p.get('source','?')}] {p.get('pmid','')} — {p.get('title','(no title)')[:100]}",
                value=True,
            )
            paper_checks.append((p["pmid"], cb))

        gr.Markdown("### Add a paper manually")
        with gr.Row():
            pmid_input = gr.Textbox(label="PMID", placeholder="e.g. 12345678", scale=1)
            add_pmid_btn = gr.Button("Look up PMID", scale=1)
        add_status = gr.Textbox(label="Add status", interactive=False)

        pdf_upload = gr.File(label="Upload PDF paper", file_types=[".pdf"])
        pdf_title = gr.Textbox(label="Paper title (for uploaded PDF)", placeholder="Enter title")
        add_pdf_btn = gr.Button("Add PDF paper")

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        result_status = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(*checkbox_values):
            excluded_pmids = set()
            for i, (pmid, _) in enumerate(paper_checks):
                if i < len(checkbox_values) and not checkbox_values[i]:
                    excluded_pmids.add(pmid)
            edited = {
                **data,
                "papers": [
                    {**p, "excluded": p["pmid"] in excluded_pmids}
                    for p in papers
                ],
                "action": "approve",
            }
            ui_handler.resume(edited)
            return "Approved. Pipeline continuing..."

        def on_add_pmid(pmid_val):
            if not pmid_val.strip():
                return "Enter a PMID first."
            return f"PMID {pmid_val.strip()} will be added when you approve."

        add_pmid_btn.click(on_add_pmid, inputs=[pmid_input], outputs=[add_status])

        cb_inputs = [cb for _, cb in paper_checks]
        approve_btn.click(on_approve, inputs=cb_inputs, outputs=[result_status])

    return panel

# slr_agent/ui/panels/synthesis.py
import gradio as gr


def build_synthesis_panel(data: dict, ui_handler) -> gr.Column:
    """Stage 6 synthesis checkpoint panel.

    data keys: synthesis_path (str), preview (str — first 500 chars of synthesis markdown)
    """
    preview = data.get("preview", "")
    synthesis_path = data.get("synthesis_path", "")

    # Try to read full synthesis if path available
    full_text = preview
    if synthesis_path:
        try:
            with open(synthesis_path) as f:
                full_text = f.read()
        except OSError:
            pass

    with gr.Column() as panel:
        gr.Markdown("## Stage 6: Synthesis\nReview the evidence synthesis below.")

        synthesis_edit = gr.Textbox(
            label="Evidence synthesis (editable)",
            value=full_text,
            lines=20,
        )

        approve_btn = gr.Button("Approve & Continue", variant="primary")
        status_out = gr.Textbox(label="Status", value="Review and approve.", interactive=False)

        def on_approve(text):
            # Save edited synthesis back to file
            if synthesis_path:
                try:
                    with open(synthesis_path, "w") as f:
                        f.write(text)
                except OSError:
                    pass
            ui_handler.resume({**data, "preview": text[:500], "action": "approve"})
            return "Approved. Pipeline continuing..."

        approve_btn.click(on_approve, inputs=[synthesis_edit], outputs=[status_out])

    return panel

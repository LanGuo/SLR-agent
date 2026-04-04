# slr_agent/ui/panels/generic.py
import gradio as gr


def build_generic_panel(stage_name: str, data: dict) -> gr.Blocks:
    """Generic JSON review panel for stages without custom UI."""
    with gr.Blocks() as panel:
        gr.Markdown(f"## Stage Review: {stage_name}")
        gr.Markdown("Review the data below. Click **Approve** to continue.")
        gr.JSON(value=data, label="Stage Output")
        approve_btn = gr.Button("Approve & Continue", variant="primary")
        status = gr.Textbox(label="Status", value="Waiting for approval...", interactive=False)
        approve_btn.click(lambda: "Approved", outputs=[status])
    return panel

# slr_agent/ui/panels/pico.py
import gradio as gr
from slr_agent.state import PICOResult


def build_pico_panel(pico: PICOResult) -> gr.Blocks:
    """Returns a Gradio Blocks panel for reviewing/editing PICO results."""
    with gr.Blocks() as panel:
        gr.Markdown("## Stage 1: PICO Formulation — Review & Edit")
        gr.Markdown("Edit any field below, then click **Approve** to continue.")
        population = gr.Textbox(label="Population (P)", value=pico["population"])
        intervention = gr.Textbox(label="Intervention (I)", value=pico["intervention"])
        comparator = gr.Textbox(label="Comparator (C)", value=pico["comparator"])
        outcome = gr.Textbox(label="Outcome (O)", value=pico["outcome"])
        queries = gr.Textbox(
            label="PubMed Query Strings (one per line)",
            value="\n".join(pico["query_strings"]),
            lines=4,
        )
        language = gr.Textbox(label="Source Language (ISO 639-1)", value=pico["source_language"])
        approve_btn = gr.Button("Approve & Continue", variant="primary")
        output = gr.JSON(label="Approved PICO", visible=False)

        def on_approve(pop, inter, comp, out, q, lang):
            result = {
                "population": pop, "intervention": inter,
                "comparator": comp, "outcome": out,
                "query_strings": [s.strip() for s in q.split("\n") if s.strip()],
                "source_language": lang,
                "search_language": "en",
                "output_language": lang,
            }
            return gr.update(visible=True, value=result)

        approve_btn.click(
            on_approve,
            inputs=[population, intervention, comparator, outcome, queries, language],
            outputs=[output],
        )
    return panel

# slr_agent/export.py
import subprocess


def run_pandoc(input_path: str, output_path: str) -> None:
    """Convert a Markdown file to .docx using Pandoc."""
    result = subprocess.run(
        ["pandoc", input_path, "-o", output_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Pandoc failed: {result.stderr}")

import json
import os
import queue
from slr_agent.emitter import ProgressEmitter


def test_emit_writes_json_file(tmp_path):
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-abc")
    emitter.emit(1, {"population": "adults", "intervention": "aspirin"})
    path = tmp_path / "run-abc" / "stage_1_pico.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["intervention"] == "aspirin"


def test_emit_calls_echo(tmp_path):
    echoed = []
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-abc", echo=echoed.append)
    emitter.emit(2, {"n_retrieved": 42})
    assert any("42" in s for s in echoed)


def test_emit_pushes_to_gradio_queue(tmp_path):
    q = queue.Queue()
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-abc", gradio_queue=q)
    emitter.emit(3, {"n_included": 10})
    msg = q.get_nowait()
    assert "10" in msg


def test_emit_creates_run_directory(tmp_path):
    emitter = ProgressEmitter(output_dir=str(tmp_path), run_id="run-xyz")
    emitter.emit(4, {"n_fetched": 0})
    assert (tmp_path / "run-xyz").is_dir()

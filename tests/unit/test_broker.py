import threading
from slr_agent.broker import CheckpointBroker, CLIHandler, UIHandler, NoOpHandler


def test_noop_handler_returns_data_unchanged():
    broker = CheckpointBroker(NoOpHandler())
    data = {"population": "adults", "intervention": "aspirin"}
    result = broker.pause(1, "pico", data)
    assert result["population"] == "adults"
    assert result["action"] == "approve"


def test_ui_handler_blocks_until_resume():
    handler = UIHandler()
    broker = CheckpointBroker(handler)
    results = []

    def pipeline():
        results.append(broker.pause(1, "pico", {"x": 1}))

    t = threading.Thread(target=pipeline)
    t.start()
    pending = handler.get_pending(timeout=2.0)
    assert pending is not None
    assert pending["stage"] == 1
    handler.resume({"x": 99, "action": "approve"})
    t.join(timeout=2.0)
    assert results[0]["x"] == 99


def test_ui_handler_get_pending_returns_none_when_empty():
    handler = UIHandler()
    assert handler.get_pending(timeout=0.05) is None


def test_cli_handler_approve(monkeypatch):
    monkeypatch.setattr("click.prompt", lambda *a, **kw: "A")
    broker = CheckpointBroker(CLIHandler())
    data = {"n": 5}
    result = broker.pause(2, "search", data)
    assert result["action"] == "approve"
    assert result["n"] == 5


def test_cli_handler_skip(monkeypatch):
    monkeypatch.setattr("click.prompt", lambda *a, **kw: "S")
    broker = CheckpointBroker(CLIHandler())
    result = broker.pause(3, "screening", {"papers": []})
    assert result["action"] == "approve"

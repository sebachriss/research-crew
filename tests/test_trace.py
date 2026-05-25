from src.ui.trace import COLOR_BY_LEVEL, render_trace_event


def test_render_trace_event_info_has_timestamp_and_node_and_text():
    event = {
        "timestamp": "2026-05-24T10:00:00",
        "node": "supervisor",
        "level": "info",
        "text": "PLAN: 3 queries",
    }
    md = render_trace_event(event)
    assert "10:00:00" in md
    assert "SUPERVISOR" in md
    assert "PLAN: 3 queries" in md


def test_render_trace_event_uses_color_by_level():
    event_info = {"timestamp": "2026-05-24T10:00:00", "node": "x", "level": "info", "text": "t"}
    event_warn = {"timestamp": "2026-05-24T10:00:00", "node": "x", "level": "warn", "text": "t"}
    event_error = {"timestamp": "2026-05-24T10:00:00", "node": "x", "level": "error", "text": "t"}

    md_info = render_trace_event(event_info)
    md_warn = render_trace_event(event_warn)
    md_error = render_trace_event(event_error)

    assert COLOR_BY_LEVEL["info"] in md_info
    assert COLOR_BY_LEVEL["warn"] in md_warn
    assert COLOR_BY_LEVEL["error"] in md_error

# tests/test_routing.py
#
# Tests for graph/routing.py — should_revise, route_after_reader,
# route_after_writer. Pure functions: plain dict in, string out. Zero LLM
# calls, zero API keys, runs in under a second.
#
# Run with:  pytest tests/test_routing.py -v
# (conftest.py handles the sys.path setup, so this works from any cwd)

from graph.routing import should_revise, route_after_reader, route_after_writer


# ---------- should_revise ----------

def test_should_revise_no_critique_yet():
    state = {"critique": None, "revision_count": 0}
    assert should_revise(state) == "pass"


def test_should_revise_pass_verdict():
    state = {"critique": {"verdict": "PASS"}, "revision_count": 0}
    assert should_revise(state) == "pass"


def test_should_revise_revise_verdict_under_cap():
    state = {"critique": {"verdict": "REVISE"}, "revision_count": 0}
    assert should_revise(state) == "revise"


def test_should_revise_revise_verdict_but_at_cap():
    # MAX_REVISIONS is 2 — at revision_count == 2 we must stop looping
    # even if the critic still says REVISE, or the graph never terminates.
    state = {"critique": {"verdict": "REVISE"}, "revision_count": 2}
    assert should_revise(state) == "pass"


# ---------- route_after_reader ----------

def test_route_after_reader_success():
    assert route_after_reader({"reader_failed": False}) == "write"


def test_route_after_reader_failure():
    assert route_after_reader({"reader_failed": True}) == "insufficient"


def test_route_after_reader_missing_key_defaults_to_write():
    assert route_after_reader({}) == "write"


# ---------- route_after_writer ----------

def test_route_after_writer_success():
    assert route_after_writer({"writer_failed": False}) == "critique"


def test_route_after_writer_failure():
    assert route_after_writer({"writer_failed": True}) == "failed"
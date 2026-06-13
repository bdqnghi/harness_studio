"""Phase 1: structured evidence model + per-harness store."""

from studio.components.evidence import (
    EvidenceStore,
    TaskEvidence,
    TraceWindow,
    VerifierSignal,
    evidence_brief,
    render_evidence_md,
    select_windows,
    to_flat_excerpt,
)


def _dialogue():
    # 8-turn dialogue; the causal failure is the wrong tool call at index 3,
    # NOT in the last 4 messages — exactly the case the old tail missed.
    return [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "change my flight"},
        {"role": "assistant", "content": "let me look"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "book_reservation", "arguments": "{}"}}]},
        {"role": "tool", "name": "book_reservation", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "thanks"},
        {"role": "assistant", "content": "bye"},
    ]


def test_select_windows_centers_on_failed_action_not_tail():
    msgs = _dialogue()
    sig = VerifierSignal(kind="action", name="update_reservation", passed=False,
                         detail="expected update_reservation, agent called book_reservation")
    # The verifier names the WRONG call site via the tool actually used.
    sig2 = VerifierSignal(kind="action", name="book_reservation", passed=False,
                          detail="should not have booked")
    windows = select_windows(msgs, [sig, sig2], task_id="t1", radius=1)
    covered = {i for w in windows for i in range(w.start_idx, w.end_idx + 1)}
    assert 3 in covered  # the wrong-tool turn is captured
    assert any("book_reservation" in (w.reason or "") for w in windows)
    # terminal turn always present
    assert (len(msgs) - 1) in covered


def test_select_windows_tail_fallback_when_nothing_localizes():
    msgs = _dialogue()
    # a db check that names nothing findable in the transcript
    sig = VerifierSignal(kind="db", name="", passed=False, detail="db mismatch")
    windows = select_windows(msgs, [sig], task_id="t1", tail=4)
    assert len(windows) == 1
    assert windows[0].start_idx == len(msgs) - 4
    assert "tail" in windows[0].reason


def test_select_windows_empty_transcript():
    assert select_windows([], [], task_id="t") == []


def test_to_flat_excerpt_includes_message_content_and_caps():
    msgs = [{"role": "assistant", "content": "wrong move"}]
    windows = select_windows(msgs, [], task_id="t2")  # no signals -> tail
    ev = TaskEvidence(task_id="t2", reward=0.0, windows=windows)
    flat = to_flat_excerpt(ev, cap=2400)
    assert "wrong move" in flat          # back-compat: content survives
    assert flat.startswith("reward=0.0")
    assert len(to_flat_excerpt(ev, cap=10)) == 10  # cap respected


def test_flat_excerpt_lists_failed_checks():
    ev = TaskEvidence(task_id="t", reward=0.0, signals=[
        VerifierSignal("action", "update_reservation", False, "expected"),
        VerifierSignal("action", "get_user_details", True, ""),  # passed -> omitted
    ])
    flat = to_flat_excerpt(ev)
    assert "update_reservation" in flat
    assert "get_user_details" not in flat  # passed checks aren't noise


def test_evidence_brief_is_budgeted():
    ev = TaskEvidence(task_id="t", reward=0.0, windows=select_windows(
        [{"role": "assistant", "content": "x" * 5000}], [], task_id="t"))
    assert len(evidence_brief(ev, cap=500)) <= 500


def test_store_versions_by_hash_and_evicts_lru():
    store = EvidenceStore()
    for i in range(EvidenceStore._MAX_HASHES + 3):
        store.put(f"h{i}", TaskEvidence(task_id="t", reward=0.0))
    assert len(store._by_hash) == EvidenceStore._MAX_HASHES
    assert store.get("h0", "t") is None          # oldest evicted
    assert store.get(f"h{EvidenceStore._MAX_HASHES + 2}", "t") is not None

    # different harness hashes don't collide
    store.put("hashA", TaskEvidence(task_id="t1", reward=1.0))
    store.put("hashB", TaskEvidence(task_id="t1", reward=0.0))
    assert store.get("hashA", "t1").reward == 1.0
    assert store.get("hashB", "t1").reward == 0.0


def test_put_refreshes_recency():
    store = EvidenceStore()
    store.put("keep", TaskEvidence(task_id="t", reward=0.0))
    for i in range(EvidenceStore._MAX_HASHES):
        store.put(f"h{i}", TaskEvidence(task_id="t", reward=0.0))
        store.put("keep", TaskEvidence(task_id="t", reward=0.0))  # touch -> stays
    assert store.get("keep", "t") is not None


def test_materialize_writes_per_task_files(tmp_path):
    store = EvidenceStore()
    ev = TaskEvidence(
        task_id="task-1", reward=0.0, transcript_len=8,
        signals=[VerifierSignal("action", "update_reservation", False, "expected X")],
        windows=[TraceWindow("task-1", 0, 2, 4,
                             [{"role": "assistant", "content": "oops"}], "action failed")],
        full_messages=[{"role": "user", "content": "hi"}],
    )
    store.put("h1", ev)
    dest = store.materialize("h1", tmp_path / "ev")
    f = dest / "task-1.md"
    assert f.is_file()
    text = f.read_text()
    assert "update_reservation" in text
    assert "Causal transcript windows" in text
    assert "Full transcript (8 messages)" in text


def test_render_evidence_md_smoke():
    ev = TaskEvidence(task_id="t", reward=1.0)
    assert "Task t" in render_evidence_md(ev)


# --- Phase 7: coding-style evidence helper (nexau / mini-swe generalization) ---

def test_evidence_from_trace_builds_signal_and_window():
    from studio.components.evidence import evidence_from_trace
    ev = evidence_from_trace(
        "write-compressor", 0.0,
        verifier_output="E FileNotFoundError: /app/out.txt\n2 failed",
        messages=[{"role": "assistant", "content": "I will run gzip"},
                  {"role": "tool", "content": "gzip: command not found"}],
    )
    assert ev.reward == 0.0
    assert any(s.kind == "test" and not s.passed and "FileNotFoundError" in s.detail
               for s in ev.signals)
    flat = to_flat_excerpt(ev)
    assert "gzip: command not found" in flat        # trajectory window survives
    assert ev.full_messages and ev.transcript_len == 2


def test_evidence_from_trace_passing_task_has_no_signal():
    from studio.components.evidence import evidence_from_trace
    ev = evidence_from_trace("t", 1.0, verifier_output="ok", messages=[])
    assert ev.signals == []

from decimal import Decimal

import pytest

from jev_photo.jev.grounding import locate
from jev_photo.jev.numeric import decode_numbers, interval_question
from jev_photo.jev.schema import Question

from .conftest import ScriptedSession


def number(instructions, lo, hi, step, **kw):
    return Question(type="number", instructions=instructions, range=[lo, hi], resolution=step, **kw)


def test_interval_cuts_cover_grid_without_gaps():
    q = number("x", "0", "1.01", "0.01")
    lo, step, cells = q.grid()
    choice, cuts = interval_question(q, lo, step, 0, cells)
    assert cuts[0] == 0 and cuts[-1] == cells == 101
    assert all(a < b for a, b in zip(cuts, cuts[1:]))
    assert len(choice.criteria) == 10


def test_interval_decoding_reaches_resolution_in_log_k_rounds(prompter, processor):
    questions = {"price": number("PRICE?", "0", "10000", "0.01"), "tilt": number("TILT?", "-50", "50", "0.5")}
    session = ScriptedSession(processor.tokenizer, interval_targets={"PRICE?": 3230.78, "TILT?": -10.0})
    out, rounds = decode_numbers(questions, prompter, session)
    assert out["price"]["exact"] == "3230.78" and out["tilt"]["value"] == -10.0
    # 1e6 cells -> 6 rounds of K=10; both questions share every round.
    assert rounds == 6 and session.calls == 6
    assert out["price"]["interval"] == ["3230.78", "3230.79"]


@pytest.mark.parametrize("lo,hi,step,written,expected", [
    ("-50", "50", "0.5", "-10", "-10"),
    ("0", "2", "0.05", "1.35", "1.35"),
    ("0", "2", "0.05", "1.33", "1.3"),     # snapped down onto the grid
    ("-1", "1", "0.05", "0", "0"),
    ("0", "10", "1", "7", "7"),
])
def test_digit_decoding_writes_then_snaps(prompter, processor, lo, hi, step, written, expected):
    q = {"v": number("VALUE?", lo, hi, step, decoding="digits")}
    session = ScriptedSession(processor.tokenizer, text_targets=[("Answer:\n", written)])
    out, _ = decode_numbers(q, prompter, session)
    assert out["v"]["written"] == written
    assert Decimal(out["v"]["exact"]) == Decimal(expected)


def test_digit_grammar_forbids_leading_zero_and_extra_places(prompter, processor):
    # "0" then only "." or end; with 2 places a third decimal is impossible.
    q = {"v": number("VALUE?", "0", "2", "0.05", decoding="digits")}
    session = ScriptedSession(processor.tokenizer, text_targets=[("Answer:\n", "0.05")])
    out, _ = decode_numbers(q, prompter, session)
    assert out["v"]["written"] == "0.05"
    # A third decimal is never offered: after two places the only legal step is a forced end.
    session = ScriptedSession(processor.tokenizer, text_targets=[("Answer:\n", "0.051")])
    out, _ = decode_numbers(q, prompter, session)
    assert out["v"]["written"] == "0.05"
    with pytest.raises(AssertionError, match="no legal token"):  # "01" would be a leading zero
        decode_numbers(q, prompter, ScriptedSession(processor.tokenizer, text_targets=[("Answer:\n", "01")]))


def test_box_walk_reads_native_bbox(prompter, processor):
    session = ScriptedSession(processor.tokenizer, text_targets=[('"bbox_2d": [', "161, 30, 385, 1000]")])
    out, steps = locate(prompter, session, {"head": "the head"})
    assert out["head"]["bbox_2d"] == [161, 30, 385, 1000]
    assert out["head"]["box"] == [0.161, 0.03, 0.385, 1.0]

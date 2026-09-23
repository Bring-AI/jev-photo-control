import pytest
from pydantic import ValidationError

from jev_photo.jev.schema import Question, Request, answer


def test_number_requires_integer_grid():
    Question(type="number", instructions="x", range=[0, 1], resolution="0.25")
    with pytest.raises(ValidationError, match="integer multiple"):
        Question(type="number", instructions="x", range=[0, 1], resolution="0.3")
    with pytest.raises(ValidationError, match="lower < upper"):
        Question(type="number", instructions="x", range=[1, 0])


def test_type_specific_fields_are_rejected_elsewhere():
    with pytest.raises(ValidationError):
        Question(type="number", instructions="x", range=[0, 1], criteria={"a": "A", "b": "B"})
    with pytest.raises(ValidationError):
        Question(type="box", instructions="x", range=[0, 1])
    with pytest.raises(ValidationError):
        Question(type="choice", instructions="x", criteria={"a": "A", "b": "B"}, decoding="digits")


def test_noul_and_score_answers():
    noul = Question(type="noul", instructions="x")
    assert answer(noul, [2.0, 0.0], 1.0)["noul"] == pytest.approx(0.8808, abs=1e-4)
    score = Question(type="score", instructions="x", criteria=["low", "mid", "high"])
    assert answer(score, [0.0, 0.0, 0.0], 1.0)["score"] == pytest.approx(1.0)


def test_request_limits():
    with pytest.raises(ValidationError):
        Request(image="x", questions={})

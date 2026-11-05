import pytest

from voiceagent.textchunk import ClauseChunker, clauses


def test_breaks_on_sentence_end():
    c = ClauseChunker()
    out = []
    for delta in ["Sure", ", I can book", " that. What day", " works for you?"]:
        out += c.push(delta)
    assert out == ["Sure, I can book that.", "What day works for you?"]
    assert c.flush() == ""


def test_decimal_point_is_not_a_sentence_break():
    c = ClauseChunker()
    out = []
    for delta in ["That will be 1", "2.50 total. Okay?"]:
        out += c.push(delta)
    assert out == ["That will be 12.50 total.", "Okay?"]


def test_soft_break_only_after_min_chars():
    c = ClauseChunker(min_chars=28)
    assert c.push("Yes, ok") == []          # too short to be worth speaking
    assert c.push(" and I have checked the calendar, ") == [
        "Yes, ok and I have checked the calendar,"
    ]


def test_runaway_sentence_is_forced_out():
    c = ClauseChunker(min_chars=28, max_chars=60)
    out = c.push("word " * 40)
    assert out, "a sentence with no punctuation must still be spoken"
    assert all(len(chunk) <= 60 for chunk in out)


def test_flush_returns_the_tail():
    c = ClauseChunker()
    assert c.push("no terminator here") == []
    assert c.flush() == "no terminator here"


async def test_clauses_wraps_an_async_delta_stream():
    async def deltas():
        for d in ["Hello", " there.", " Bye"]:
            yield d

    assert [c async for c in clauses(deltas())] == ["Hello there.", "Bye"]

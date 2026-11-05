from voiceagent.audio import FRAME_BYTES, FRAME_MS
from voiceagent.vad import SpeechGate, TurnDetector
from conftest import silence, tone


def frames(pcm: bytes):
    return [pcm[i : i + FRAME_BYTES] for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)]


def test_gate_hysteresis_holds_through_a_dip():
    gate = SpeechGate(open_threshold=0.02, close_threshold=0.01)
    # A level between the two thresholds cannot open the gate...
    assert gate.is_speech(tone(20, amplitude=0.022)) is False
    # ...but once open, that same level keeps it open.
    assert gate.is_speech(tone(20, amplitude=0.4)) is True
    assert gate.is_speech(tone(20, amplitude=0.022)) is True
    assert gate.is_speech(silence(20)) is False


def test_turn_detector_reports_start_then_end():
    det = TurnDetector(end_of_turn_silence_ms=200)
    events = [det.feed(f, FRAME_MS) for f in frames(tone(500) + silence(400))]
    seen = [e for e in events if e]
    assert seen == ["speech_start", "speech_end"]


def test_short_silence_does_not_end_the_turn():
    """A pause mid-sentence must not hand the turn over early."""
    det = TurnDetector(end_of_turn_silence_ms=640)
    seq = tone(400) + silence(300) + tone(400) + silence(700)
    seen = [e for e in (det.feed(f, FRAME_MS) for f in frames(seq)) if e]
    assert seen == ["speech_start", "speech_end"]


def test_isolated_noise_never_starts_a_turn():
    det = TurnDetector(min_speech_ms=160)
    seq = b"".join(tone(20) + silence(100) for _ in range(12))
    assert all(e is None for e in (det.feed(f, FRAME_MS) for f in frames(seq)))
    assert det.speaking is False


def test_reset_clears_state():
    det = TurnDetector(end_of_turn_silence_ms=200)
    for f in frames(tone(400)):
        det.feed(f, FRAME_MS)
    assert det.speaking
    det.reset()
    assert not det.speaking

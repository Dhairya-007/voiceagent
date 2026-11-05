from voiceagent.audio import (
    FRAME_BYTES, PCM_RATE, Framer, Resampler, TelephonyCodec, bytes_to_ms, rms,
)
from conftest import silence, tone


def test_frame_size_is_20ms():
    assert FRAME_BYTES == 640
    assert bytes_to_ms(FRAME_BYTES) == 20.0


def test_framer_emits_uniform_frames_and_keeps_remainder():
    f = Framer(FRAME_BYTES)
    assert f.push(b"\x00" * (FRAME_BYTES + 100)) == [b"\x00" * FRAME_BYTES]
    assert len(f.flush()) == 100


def test_framer_reassembles_across_odd_chunk_sizes():
    f = Framer(FRAME_BYTES)
    frames = []
    for _ in range(100):
        frames += f.push(b"\x01" * 137)
    # 13700 bytes in -> 21 whole frames (13440 bytes), 260 bytes left over.
    assert all(len(fr) == FRAME_BYTES for fr in frames)
    assert len(frames) == 21
    assert len(f.flush()) == 13700 - 21 * FRAME_BYTES


def test_telephony_codec_roundtrip_preserves_duration():
    codec = TelephonyCodec()
    # 20 ms of mu-law at 8 kHz is 160 bytes; decoded it is 20 ms at 16 kHz.
    out = b"".join(codec.decode(b"\xff" * 160) for _ in range(50))
    expected = 50 * 640
    # Allow the single sample the interpolator consumes priming itself.
    assert abs(len(out) - expected) <= 4


def test_stateful_resampler_loses_far_less_than_stateless():
    """Regression test for the clicking bug.

    Calling audioop.ratecv with state=None per chunk drops a sample at every
    boundary; a stateful resampler pays that cost once.
    """
    chunks = [b"\x00\x10" * 160] * 20
    stateful = Resampler(8_000, PCM_RATE)
    got = sum(len(stateful(c)) for c in chunks)

    stateless = sum(len(Resampler(8_000, PCM_RATE)(c)) for c in chunks)

    assert got > stateless
    assert abs(got - 20 * 640) <= 4
    # The stateless version is short by roughly one sample per chunk.
    assert 20 * 640 - stateless >= 2 * len(chunks) - 4


def test_resampler_is_noop_at_matching_rates():
    r = Resampler(PCM_RATE, PCM_RATE)
    payload = b"\x01\x02\x03\x04"
    assert r(payload) is payload


def test_rms_separates_speech_from_silence():
    assert rms(silence(20)) == 0.0
    assert rms(tone(20)) > 0.2
    assert rms(b"") == 0.0

# voiceagent

A low-latency streaming voice agent for telephony: **speech in → transcript →
LLM → speech out**, with barge-in, turn-taking and per-turn latency
instrumentation.

It answers a browser microphone or a real phone number over Twilio Media
Streams, handles being interrupted mid-sentence, and records a latency trace
for every turn.

**696 ms p50 response latency** on a laptop CPU against an 800 ms budget,
measured end to end from speech-end to first audio out.

Runs entirely on free, local components. The default configuration requires no
API key and no network.

---

## Demo

**[demo/demo.mp4](demo/demo.mp4)** — a complete recorded call. Transcripts come
from Whisper running locally; every latency figure is measured from that run.

Reproduce it:

```bash
python scripts/record_demo.py --out demo --stt whisper --tts espeak
```

```bash
python scripts/render_video.py --demo demo --out demo/demo.mp4
```

## Quick start

```bash
./run.sh
```

Open <http://localhost:8080>, press **Start call**, and speak. Interrupting the
agent mid-sentence demonstrates barge-in. The right-hand panel plots the
latency waterfall per turn as the call proceeds.

The default configuration uses in-process mock providers, so this works with no
installation and no network access.

## Design

A caller begins to perceive a gap at roughly 500 ms and will talk over the
agent beyond about one second. That budget must cover transcription, response
generation and synthesis — which is only achievable if those stages overlap
rather than run in sequence.

```
  caller audio  ──►  Framer  ──►  TurnDetector ──► speech_start / speech_end
   (mic | PSTN)      20 ms          VAD +
                     frames       hysteresis
                                       │
         ┌─────────────────────────────┼──────────────────────────┐
         ▼                             ▼                          ▼
     LISTENING                     THINKING                   SPEAKING
   stream audio                 STT final text              stream TTS out,
   to STT                       → LLM (streaming)           monitor barge-in
                                → clause chunker                  │
                                → TTS (streaming)                 │
                                                        barge-in: cancel task,
                                                        clear carrier buffer
```

Three properties make the interaction feel conversational:

**Stages overlap.** STT streams while the caller is still speaking. LLM output
passes through a clause chunker ([`textchunk.py`](src/voiceagent/textchunk.py))
so synthesis begins on the first clause while the model is still generating the
rest. Time-to-first-audio becomes `first-clause LLM + first-clause TTS` rather
than `full LLM + full TTS`.

**A turn is a single cancellable task.** Barge-in is implemented as task
cancellation plus a buffer flush, not a flag polled downstream. On a carrier
leg it additionally issues Twilio's `clear` event, because the carrier buffers
audio that local cancellation does not affect.

**Latency is measured per turn.** Each turn records absolute marks, so the
distribution of a given response time across stages is derivable from data
already persisted.

### Module responsibilities

| Module | Responsibility |
|---|---|
| [`orchestrator.py`](src/voiceagent/orchestrator.py) | Call state machine, barge-in, cancellation |
| [`audio.py`](src/voiceagent/audio.py) | Framing, mu-law transcoding, stateful resampling |
| [`vad.py`](src/voiceagent/vad.py) | Energy gate with hysteresis, end-of-turn detection |
| [`textchunk.py`](src/voiceagent/textchunk.py) | LLM deltas to speakable clauses |
| [`metrics.py`](src/voiceagent/metrics.py) | Per-turn marks, latency breakdown, anomaly detection |
| [`telephony/twilio.py`](src/voiceagent/telephony/twilio.py) | Media Streams, 8 kHz mu-law, `clear` |
| [`providers/`](src/voiceagent/providers/) | One protocol per stage; vendor adapters behind it |

The orchestrator imports no vendor SDK. Replacing Whisper with Deepgram, or
espeak-ng with a neural voice, affects one file and no call logic.

## Performance

Measured from the recorded call in `demo/` — faster-whisper `tiny.en`, a
deterministic rule-based responder, and espeak-ng, all local on CPU:

| Turn | STT finalize | LLM first token | TTS first byte | Total |
|-----:|-------------:|----------------:|---------------:|------:|
| 1 | 268 ms | 261 ms | 193 ms | **722 ms** |
| 2 | 394 ms | 263 ms | 66 ms | **722 ms** |
| 3 | 245 ms | 262 ms | 188 ms | **696 ms** |
| 4 | 262 ms | 262 ms | 100 ms | **623 ms** |

p50 **696 ms** · p95 **722 ms** · 4/4 turns within budget · 1 barge-in.

### Optimisations

**Synthesiser process model.** Measured time-to-first-byte differs by two
orders of magnitude between free engines:

```
macOS say   ~860 ms   constant, independent of input length
espeak-ng    ~15 ms
```

A cost independent of input length indicates initialisation, not computation —
`say` reloads its voice on every invocation. Adopting a streaming synthesiser
reduced p50 from 1488 ms to 696 ms. The same principle governs
[`tts_piper.py`](src/voiceagent/providers/tts_piper.py), which loads its model
once and retains it in process; invoking the equivalent CLI per utterance would
reintroduce the same cost.

**Interim transcript scheduling.** Whisper is not a streaming model, so interim
results require re-decoding a growing buffer. Those decodes contended with the
end-of-turn decode on which response latency actually depends. Interim requests
are now dropped while a decode is in flight, reducing p50 by approximately
900 ms.

**Latency anomaly detection.** `TurnTrace.anomalies()` reports out-of-order
marks rather than emitting negative durations, so instrumentation faults
surface instead of being averaged into percentiles.

## Configuration

Every supported provider is free. See [`.env.example`](.env.example).

| Stage | Provider | Notes |
|---|---|---|
| STT | `faster-whisper` | Local, offline. `tiny.en` is ~75 MB |
| LLM | Ollama, llama.cpp, vLLM | Local. Groq's free tier uses the same protocol |
| TTS | espeak-ng | ~15 ms to first byte |
| TTS | Piper | Neural voice, loaded once in process |
| TTS | macOS `say` | Natural voice, ~860 ms per utterance |

```bash
brew install espeak-ng && uv pip install faster-whisper
```

```bash
STT_PROVIDER=whisper TTS_PROVIDER=espeak ./run.sh
```

The default LLM is a deterministic rule-based responder, so the repository runs
without network access. Set `LLM_PROVIDER=openai_compat` and point
`LLM_BASE_URL` at Ollama or any OpenAI-compatible endpoint to use a model.

A provider that fails to initialise degrades to its mock and the call continues;
`/healthz` reports configured versus resolved providers, which distinguishes a
misconfigured deployment from a failing one.

### Telephony

```bash
PUBLIC_WS_URL=wss://your-host/telephony/twilio/stream ./run.sh
```

Point the number's voice webhook at `POST /telephony/twilio/voice`. Twilio is
the only non-free dependency and is opt-in; the browser demo does not use it.

## Implementation notes

**Resampler state.** `audioop.ratecv` maintains interpolation state across
calls. Passing `None` per chunk drops a sample at every chunk boundary,
producing a periodic artefact at the frame rate. Covered by a regression test
in [`tests/test_audio.py`](tests/test_audio.py).

**Echo-induced false barge-in.** The caller's microphone captures the agent's
own output over a speakerphone, or as line echo on a carrier leg. At the
listening threshold this registers as barge-in and the agent interrupts itself.
Mitigated by raising the VAD threshold during playout; acoustic echo
cancellation is the complete solution.

**End-of-turn sensitivity.** A detector that ends the turn on first silence
will cut the caller off mid-sentence. Hysteresis combined with a
trailing-silence window (`END_OF_TURN_SILENCE_MS`, default 640 ms) governs this,
and is the parameter with the largest effect on perceived responsiveness.

**Audio thread isolation.** Capture and playback run in AudioWorklets rather
than on the main thread, so UI work cannot drop audio frames. Blocking provider
calls run via `asyncio.to_thread`, keeping the inbound audio pump — and
therefore barge-in detection — responsive.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

35 tests. Integration tests exercise the orchestrator, VAD, clause chunker and
metrics against fake providers and a fake transport, covering a complete turn,
barge-in cancelling playback, echo not triggering false barge-in, mid-sentence
pauses not ending a turn, and provider degradation.

## Limitations

- **Energy-based VAD** performs adequately on clean audio and degrades in noisy
  environments. `SpeechGate.is_speech` is the single replacement point for
  webrtcvad or Silero; the surrounding hysteresis is unchanged.
- **Whisper interim results may revise themselves**, as it re-decodes rather
  than streams. A streaming model avoids this without interface changes.
- **No acoustic echo cancellation.** The browser relies on the WebRTC stack; a
  carrier leg requires dedicated AEC.
- **Piper's macOS wheel** can abort during phonemisation with a build-time
  espeak path. The process aborts rather than raising, and only after model load
  succeeds, so `build_tts` probes it out of process and falls back.

## Requirements

Python 3.10+. `audioop` was removed from the standard library in Python 3.13;
the `audioop-lts` backport is declared conditionally.

## License

MIT — see [LICENSE](LICENSE).

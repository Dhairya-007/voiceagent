# voiceagent

A streaming voice agent for phone calls: **speech in → transcript → LLM → speech
out**, built so the caller does not notice the machinery. It answers a browser
mic or a real phone number, handles being interrupted mid-sentence, and records
a latency trace for every turn.

Every component is free. The default configuration needs no account, no API key
and no network.

**p50 response latency 696 ms** on a laptop CPU, against an 800 ms budget —
measured end to end, speech-end to first audio out.

## Demo

`demo/demo.mp4` — one real recorded call. Both voices are synthesized, the
transcripts come from Whisper running locally, and every latency bar is measured
from that run. Nothing on screen is mocked up.

Regenerate it yourself:

```bash
./run.sh                    # the browser demo at http://localhost:8080
```

```bash
python scripts/record_demo.py --out demo --stt whisper --tts espeak
```

```bash
python scripts/render_video.py --demo demo --out demo/demo.mp4
```

## The problem this is built around

A caller starts to feel a gap at roughly **500 ms** and starts talking over you
past about a second. That budget has to cover transcribing what they said,
deciding what to say back, and synthesizing it — so the only way to fit is to
stop treating those as three sequential steps.

Three things make the difference, and the code is organized around them:

**1. Everything overlaps.** STT streams while the caller is still talking. The
LLM's output goes through a clause chunker ([`textchunk.py`](src/voiceagent/textchunk.py))
so the synthesizer starts speaking *"Sure, I can book that."* while the model is
still writing *"What day works for you?"*. Naively, time-to-first-audio is
`full LLM + full TTS`; chunked, it is `first-clause LLM + first-clause TTS`.

**2. A turn is one cancellable task.** Barge-in is `task.cancel()` plus a flush —
not a flag polled somewhere downstream. On a real call it also has to send
Twilio a `clear`, because the carrier has already buffered a second of speech
that cancelling the synthesizer locally does nothing about.

**3. Latency is measured per turn, not per service.** Each turn records absolute
marks, so "where did the 700 ms go" is answered from data already on disk.

## Measured results

From the recorded call in `demo/` (faster-whisper `tiny.en` + rule-based LLM +
espeak-ng, all local on CPU):

| turn | stt finalize | llm 1st token | tts 1st byte | **total** |
|-----:|-------------:|--------------:|-------------:|----------:|
| 2 | 268 ms | 261 ms | 193 ms | **722 ms** |
| 3 | 394 ms | 263 ms | 66 ms | **722 ms** |
| 4 | 245 ms | 262 ms | 188 ms | **696 ms** |
| 5 | 262 ms | 262 ms | 100 ms | **623 ms** |

p50 **696 ms** · p95 **722 ms** · 4/4 turns within budget · 1 barge-in.

The `llm 1st token` column is a rule-based stub with a deliberate 260 ms delay
standing in for a real model — it is honest about being a stand-in, not a claim
about model speed.

### Two things measurement caught

**TTS process spawn dominates everything else.** Rendering a clause with macOS
`say`:

```
say         ~860 ms to first byte   — flat, regardless of text length
espeak-ng    ~15 ms to first byte
```

`say`'s 860 ms is voice loading, paid on every spawn, and it blows the entire
budget before the synthesizer emits a sample. Switching to espeak-ng took p50
from 1488 ms to 696 ms. The general lesson is in
[`tts_piper.py`](src/voiceagent/providers/tts_piper.py): a neural voice *held
warm in-process* gets both quality and latency; the same voice spawned per
utterance gets neither.

**Interim transcripts were delaying the final one.** Whisper is not a streaming
model, so interim results come from re-decoding a growing buffer. Those decodes
queued ahead of the end-of-turn decode — the one the caller is actually waiting
on. Dropping an interim when a decode is already in flight took ~900 ms off p50.

## Architecture

```
  caller audio  ──►  Framer  ──►  TurnDetector ──► speech_start / speech_end
   (mic | PSTN)      20 ms          VAD +
                     frames       hysteresis
                                       │
         ┌─────────────────────────────┼──────────────────────────┐
         ▼                             ▼                          ▼
     LISTENING                     THINKING                   SPEAKING
   stream audio                 STT final text              stream TTS out,
   to STT                       → LLM (streaming)           watch for barge-in
                                → clause chunker                  │
                                → TTS (streaming)                 │
                                                        barge-in: cancel task,
                                                        clear carrier buffer
```

| file | what it owns |
|---|---|
| [`orchestrator.py`](src/voiceagent/orchestrator.py) | the state machine, barge-in, cancellation |
| [`audio.py`](src/voiceagent/audio.py) | framing, mu-law, **stateful** resampling |
| [`vad.py`](src/voiceagent/vad.py) | energy gate with hysteresis, end-of-turn |
| [`textchunk.py`](src/voiceagent/textchunk.py) | LLM deltas → speakable clauses |
| [`metrics.py`](src/voiceagent/metrics.py) | per-turn marks, latency breakdown, anomalies |
| [`telephony/twilio.py`](src/voiceagent/telephony/twilio.py) | Media Streams, 8 kHz mu-law, `clear` |
| [`providers/`](src/voiceagent/providers/) | one protocol per stage; vendors behind it |

The orchestrator never imports a vendor SDK. Swapping Whisper for Deepgram, or
espeak for a neural voice, touches one file and no call logic.

## Three details that only show up on real calls

**Resampler state.** `audioop.ratecv` carries interpolation state between calls.
Passing `None` each time — the obvious way to write it — silently drops a sample
at every chunk boundary and puts a click in the stream every 20 ms. There is a
[regression test](tests/test_audio.py) for it.

**The agent interrupting itself.** The caller's mic hears the agent, over a
speakerphone or as line echo. At the listening threshold, the agent's own voice
reads as a barge-in and it cuts itself off mid-sentence. Mitigated here by
raising the VAD threshold during playout; the real fix is acoustic echo
cancellation.

**Mid-sentence pauses.** A VAD that ends the turn on the first silence hands the
turn over while the caller is drawing breath. Hysteresis plus a trailing-silence
window (`END_OF_TURN_SILENCE_MS`, default 640 ms) is what stops that, and it is
the single knob that most changes how the agent feels.

## Running it

```bash
./run.sh
```

Open <http://localhost:8080>, press **Start call**, and talk. Interrupt the
agent mid-sentence to watch barge-in cancel playback. The right-hand panel draws
the latency waterfall per turn as the call runs.

Out of the box it runs on mocks — no install, no network. To make it real, see
[`.env.example`](.env.example); every option there is free:

| stage | free option | notes |
|---|---|---|
| STT | `faster-whisper` | local, offline. `tiny.en` is ~75 MB |
| LLM | Ollama, llama.cpp, vLLM | local. Or Groq's free tier — same OpenAI protocol |
| TTS | espeak-ng | ~15 ms to first byte, robotic |
| TTS | Piper | free neural voice, loaded once in-process |
| TTS | macOS `say` | natural, but ~860 ms per utterance |

```bash
brew install espeak-ng && uv pip install faster-whisper
```

```bash
STT_PROVIDER=whisper TTS_PROVIDER=espeak ./run.sh
```

A provider that fails to start **degrades to the mock and keeps the call up**
rather than dropping it — `/healthz` reports what actually loaded, which is the
most useful line in it when a deploy "works" but sounds wrong.

### Answering a real phone number

Twilio is the only paid piece, so it is opt-in and the browser demo never
touches it.

```bash
PUBLIC_WS_URL=wss://your-tunnel.ngrok.app/telephony/twilio/stream ./run.sh
```

Point the number's voice webhook at `POST /telephony/twilio/voice`.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

35 tests. The integration tests run the real orchestrator, VAD, chunker and
metrics against fake vendors and a fake transport — the bugs in a voice agent
live in turn-taking and cancellation, not in vendor SDKs. They cover a full turn
end to end, barge-in cancelling playback, echo *not* triggering a false
barge-in, mid-sentence pauses not ending a turn, and two regressions found while
building this (see below), plus the degradation path — every provider that
cannot start must fall back to a mock rather than drop the call.

## Known issues

- **Energy VAD** is good enough on clean audio and will struggle in a noisy
  room. `SpeechGate.is_speech` is the single function to replace with webrtcvad
  or Silero; the hysteresis around it is what matters and would not change.
- **Whisper interims can rewrite themselves**, because it re-decodes a growing
  buffer rather than streaming. A streaming open model avoids this; the provider
  interface does not change.
- **No acoustic echo cancellation.** The browser leans on the WebRTC stack's
  own AEC; a PSTN leg would need real AEC rather than a raised threshold.
- **Piper's macOS wheel** can abort with a build-machine espeak path
  (`/Users/runner/.../phontab`). It aborts the *process* rather than raising,
  and only once phonemization runs — import and model load both succeed — so
  `build_tts` probes it by synthesizing one word out of process, then falls
  back. Caught at startup instead of on the first caller.

## Two bugs worth keeping

**The greeting ate the first caller turn.** Turn 1 reported a *negative*
response latency. The greeting created a metrics trace and never set
`speech_end` on it, so the first caller utterance wrote its marks onto a trace
that already had `playout_start` from the greeting — and `speech_end` landed
after it. Fixed by giving traces a `kind` and only letting an in-progress caller
turn absorb speech marks. `TurnTrace.anomalies()` now reports out-of-order marks
instead of averaging nonsense into p50, and a test asserts no turn ever reports
one.

**A test that passed for the wrong reason.** The regression test above waited
for `turn_complete` — and got the *greeting's*, which fires before the caller
has said anything, so it asserted against a turn that had not happened yet.
Waiting for a specific turn number fixed it. A test that races ahead of the
thing it is testing is worse than no test.

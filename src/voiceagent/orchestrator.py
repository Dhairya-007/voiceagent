"""The call state machine: who is allowed to talk, and when.

                  caller audio
                       |
                  [ Framer ]  uniform 20 ms frames
                       |
                 [ TurnDetector ]  speech_start / speech_end
                       |
   +-------------------+--------------------+
   |                   |                    |
LISTENING          THINKING             SPEAKING
   |                   |                    |
 forward audio      STT final ->         stream TTS audio out,
 to STT             LLM -> clauses       watch for barge-in
                    -> TTS

Three things make this feel like a phone call rather than a walkie-talkie:

1. Everything overlaps. STT streams while the caller talks, the LLM streams
   into the clause chunker, and TTS starts on clause one. Nothing waits for the
   stage before it to finish.
2. The agent turn is a single cancellable Task. Barge-in is `task.cancel()`
   plus a flush of buffered audio -- not a flag checked somewhere down the line.
3. Inbound audio is never dropped. Even mid-barge-in the frames keep flowing to
   the VAD, so the interrupting words are not clipped.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

from .audio import FRAME_BYTES, FRAME_MS, Framer, bytes_to_ms
from .config import Settings
from .metrics import CallMetrics, TurnTrace
from .providers import Message, build_llm, build_stt, build_tts, describe
from .textchunk import clauses
from .vad import SpeechGate, TurnDetector

log = logging.getLogger(__name__)


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class Transport(Protocol):
    """What the orchestrator needs from a call leg.

    Implemented by the browser websocket and by the Twilio Media Streams
    handler. The orchestrator knows nothing about either.
    """

    async def send_audio(self, pcm16k: bytes) -> None: ...
    async def send_event(self, event: dict[str, Any]) -> None: ...
    async def clear_audio(self) -> None:
        """Drop audio already handed to the transport but not yet heard.

        Essential for barge-in: Twilio buffers what you send it, so cancelling
        the synthesizer is not enough -- you must also tell the carrier to throw
        away what is queued, or the agent keeps talking for another second.
        """


class CallSession:
    def __init__(
        self,
        transport: Transport,
        settings: Settings,
        call_id: str | None = None,
        greeting: str | None = "Thanks for calling the clinic. How can I help?",
    ) -> None:
        self.transport = transport
        self.settings = settings
        self.call_id = call_id or uuid.uuid4().hex[:12]
        self.greeting = greeting

        self.stt = build_stt(settings)
        self.llm = build_llm(settings)
        self.tts = build_tts(settings)
        self.providers = describe(self.stt, self.llm, self.tts)

        self.metrics = CallMetrics(self.call_id)
        self.history: list[Message] = []
        self.state = State.IDLE

        self._framer = Framer(FRAME_BYTES)
        self._turn_detector = TurnDetector(
            end_of_turn_silence_ms=settings.end_of_turn_silence_ms
        )
        # A second, less sensitive gate used only while the agent is speaking.
        # The caller's mic hears the agent (speakerphone, or line echo on PSTN),
        # so the same threshold that works when the line is quiet will trigger
        # on the agent's own voice and the bot will interrupt itself. Raising
        # the bar during playout is the cheap fix; real AEC is the proper one.
        self._barge_gate = SpeechGate(open_threshold=0.055, close_threshold=0.035)
        self._barge_speech_ms = 0.0

        self._agent_task: asyncio.Task | None = None
        self._stt_task: asyncio.Task | None = None
        self._closed = False

    # ---------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        await self.stt.start()
        self._stt_task = asyncio.create_task(
            self._drain_transcripts(), name=f"stt-{self.call_id}"
        )
        self.state = State.LISTENING
        await self._emit("call_started", providers=self.providers)
        if self.greeting:
            # The greeting is a turn with no caller speech: it has no
            # response_latency (there was nothing to respond to), but it does
            # warm every provider's connection so turn one is not the slowest.
            await self._speak_greeting(self.greeting)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._cancel_agent_turn()
        for task in (self._stt_task,):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        for provider in (self.stt, self.llm, self.tts):
            with contextlib.suppress(Exception):
                await provider.close()
        await self._emit("call_ended", summary=self.metrics.summary())

    # ------------------------------------------------------------ inbound audio
    async def on_audio(self, pcm16k: bytes) -> None:
        """Entry point for caller audio. Must stay cheap: it runs per frame."""
        if self._closed:
            return
        for frame in self._framer.push(pcm16k):
            await self._on_frame(frame)

    async def _on_frame(self, frame: bytes) -> None:
        if self.state == State.SPEAKING:
            await self._check_barge_in(frame)
            # Deliberately not forwarded to STT: while the agent is talking the
            # inbound frames are mostly echo of the agent, and feeding them in
            # pollutes the next transcript.
            return

        self._barge_speech_ms = 0.0
        event = self._turn_detector.feed(frame, FRAME_MS)

        if self.state == State.LISTENING:
            with contextlib.suppress(Exception):
                await self.stt.send_audio(frame)

        if event == "speech_start":
            # Only an in-progress caller turn may absorb these marks. The
            # greeting trace must not: it already has playout marks, and
            # reusing it produced negative response latencies.
            trace = self.metrics.current
            if trace is None or not trace.open_for_speech:
                trace = self.metrics.new_turn(self.providers)
            trace.mark("speech_start")
            await self._emit("speech_start", turn=trace.turn)
        elif event == "speech_end":
            await self._on_speech_end()

    async def _check_barge_in(self, frame: bytes) -> None:
        if self._barge_gate.is_speech(frame):
            self._barge_speech_ms += FRAME_MS
        else:
            self._barge_speech_ms = max(0.0, self._barge_speech_ms - FRAME_MS)

        if self._barge_speech_ms >= self.settings.barge_in_speech_ms:
            self._barge_speech_ms = 0.0
            trace = self.metrics.current
            if trace:
                trace.mark("barge_in")
                trace.barged_in = True
            log.info("call=%s barge-in", self.call_id)
            await self._cancel_agent_turn()
            await self.transport.clear_audio()
            self._turn_detector.reset()
            self.state = State.LISTENING
            await self._emit("barge_in", turn=trace.turn if trace else None)

    async def _on_speech_end(self) -> None:
        trace = self.metrics.current
        if trace is None or not trace.open_for_speech:
            trace = self.metrics.new_turn(self.providers)
            trace.mark("speech_start")
        trace.mark("speech_end")
        self.state = State.THINKING
        await self._emit("speech_end", turn=trace.turn)

        # Whisper and the mock both need a nudge to emit the final; a truly
        # streaming vendor sends it unprompted and this is a no-op.
        finalize = getattr(self.stt, "finalize", None)
        if finalize is not None:
            with contextlib.suppress(Exception):
                await finalize()

    # ------------------------------------------------------------- transcripts
    async def _drain_transcripts(self) -> None:
        try:
            async for tr in self.stt.transcripts():
                trace = self.metrics.current
                if trace is None:
                    continue
                if not tr.is_final:
                    trace.mark("stt_partial")
                    await self._emit("transcript", text=tr.text, final=False,
                                     turn=trace.turn)
                    continue

                trace.mark("stt_final")
                trace.transcript = tr.text
                await self._emit("transcript", text=tr.text, final=True,
                                 turn=trace.turn, confidence=tr.confidence)
                if tr.text.strip():
                    await self._start_agent_turn(trace, tr.text)
                else:
                    # Nothing recognised -- do not burn an LLM call on silence.
                    self.state = State.LISTENING
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("call=%s transcript loop died", self.call_id)

    # ------------------------------------------------------------- agent turn
    async def _start_agent_turn(self, trace: TurnTrace, user_text: str) -> None:
        await self._cancel_agent_turn()
        self._agent_task = asyncio.create_task(
            self._run_agent_turn(trace, user_text), name=f"turn-{trace.turn}"
        )

    async def _cancel_agent_turn(self) -> None:
        task, self._agent_task = self._agent_task, None
        cancel = getattr(self.tts, "cancel", None)
        if cancel is not None:
            with contextlib.suppress(Exception):
                await cancel()
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    async def _run_agent_turn(self, trace: TurnTrace, user_text: str) -> None:
        self.history.append(Message(role="user", content=user_text))
        self._trim_history()
        reply_parts: list[str] = []

        async def deltas() -> AsyncIterator[str]:
            async for piece in self.llm.complete(
                self.settings.system_prompt, list(self.history)
            ):
                trace.mark("llm_first_token")
                reply_parts.append(piece)
                yield piece
            trace.mark("llm_done")

        async def spoken() -> AsyncIterator[str]:
            async for clause in clauses(deltas()):
                await self._emit("agent_text", text=clause, turn=trace.turn)
                yield clause

        try:
            self.state = State.SPEAKING
            self._barge_gate.reset()
            self._barge_speech_ms = 0.0
            await self._pump_audio(self.tts.synthesize(spoken()), trace)
        except asyncio.CancelledError:
            trace.reply = "".join(reply_parts) + " [interrupted]"
            raise
        except Exception:
            log.exception("call=%s turn=%s failed", self.call_id, trace.turn)
            await self._emit("error", turn=trace.turn,
                             message="agent turn failed; recovering")
        finally:
            trace.reply = trace.reply or "".join(reply_parts)
            if reply_parts and not trace.barged_in:
                self.history.append(
                    Message(role="assistant", content="".join(reply_parts))
                )
            if self.state == State.SPEAKING:
                self.state = State.LISTENING
            self._turn_detector.reset()
            await self._emit(
                "turn_complete", turn=trace.turn, trace=trace.to_dict()
            )

    async def _pump_audio(
        self, audio: AsyncIterator[bytes], trace: TurnTrace
    ) -> None:
        """Write synthesized audio to the transport, pacing it to real time.

        Pacing matters. Without it the whole reply is dumped into the socket in
        a few milliseconds, and then a barge-in has nothing left to cancel --
        the caller is already committed to hearing all of it. Writing at roughly
        the rate it will be played keeps the cancel window open.
        """
        sent_ms = 0.0
        t_start: float | None = None
        async for chunk in audio:
            if not chunk:
                continue
            trace.mark("tts_first_byte")
            if t_start is None:
                t_start = time.monotonic()
                trace.mark("playout_start")
                await self._emit("playout_start", turn=trace.turn)

            await self.transport.send_audio(chunk)
            sent_ms += bytes_to_ms(len(chunk))
            trace.audio_ms = sent_ms

            # Stay at most ~200 ms ahead of real time.
            ahead = sent_ms / 1000 - (time.monotonic() - t_start)
            if ahead > 0.2:
                await asyncio.sleep(ahead - 0.2)

        trace.mark("playout_end")
        await self._emit("playout_end", turn=trace.turn, audio_ms=round(sent_ms, 1))

    async def _speak_greeting(self, text: str) -> None:
        trace = self.metrics.new_turn(self.providers, kind="greeting")
        trace.reply = text

        async def one() -> AsyncIterator[str]:
            await self._emit("agent_text", text=text, turn=trace.turn)
            yield text

        self.state = State.SPEAKING
        try:
            await self._pump_audio(self.tts.synthesize(one()), trace)
        finally:
            self.state = State.LISTENING
            self.history.append(Message(role="assistant", content=text))
            await self._emit("turn_complete", turn=trace.turn,
                             trace=trace.to_dict())

    # ------------------------------------------------------------------ helpers
    def _trim_history(self) -> None:
        """Keep the prompt bounded.

        Prompt length is latency on every single turn, so history is capped at
        a few exchanges rather than left to grow for the length of the call.
        """
        limit = self.settings.max_history_turns * 2
        if len(self.history) > limit:
            del self.history[: len(self.history) - limit]

    async def _emit(self, type_: str, **fields: Any) -> None:
        with contextlib.suppress(Exception):
            await self.transport.send_event(
                {"type": type_, "call_id": self.call_id,
                 "state": self.state.value, **fields}
            )

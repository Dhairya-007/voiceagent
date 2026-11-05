"""Twilio Media Streams transport.

Twilio is the only non-free component in this repo, so it is opt-in and the
browser demo does not touch it. The adapter is here because a voice agent that
cannot answer a real phone number is a chatbot with a microphone, and the
interesting problems live on this side: 8 kHz mu-law, carrier buffering, and
`clear`.

Protocol, inbound over the websocket:
    connected / start / media / stop   (JSON text frames, media is b64 mu-law)
Outbound:
    {"event":"media","streamSid":...,"media":{"payload": b64}}
    {"event":"clear","streamSid":...}   <-- drop what the carrier has buffered
    {"event":"mark","streamSid":...}    <-- playback progress callback

The `clear` event is the whole reason barge-in works on a real call. Twilio
accepts audio faster than real time and buffers it; cancelling the synthesizer
locally does nothing for the second of speech already sitting in Twilio's
jitter buffer. Without `clear`, the caller interrupts and the agent keeps
talking over them.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

from ..audio import TelephonyCodec, b64, unb64
from ..config import Settings
from ..orchestrator import CallSession

log = logging.getLogger(__name__)


def twiml(public_ws_url: str) -> str:
    """TwiML to point a Twilio number at this server."""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f'<Connect><Stream url="{public_ws_url}" /></Connect>'
        "</Response>"
    )


class TwilioTransport:
    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self.stream_sid: str | None = None
        self.codec = TelephonyCodec()
        self._marks = 0

    async def send_audio(self, pcm16k: bytes) -> None:
        if not self.stream_sid:
            return
        await self.ws.send_text(
            json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": b64(self.codec.encode(pcm16k))},
            })
        )

    async def clear_audio(self) -> None:
        if not self.stream_sid:
            return
        await self.ws.send_text(
            json.dumps({"event": "clear", "streamSid": self.stream_sid})
        )

    async def send_mark(self, name: str) -> None:
        """Ask Twilio to tell us when playback reaches this point."""
        if not self.stream_sid:
            return
        self._marks += 1
        await self.ws.send_text(
            json.dumps({
                "event": "mark",
                "streamSid": self.stream_sid,
                "mark": {"name": f"{name}-{self._marks}"},
            })
        )

    async def send_event(self, event: dict[str, Any]) -> None:
        # Twilio's socket carries audio only; app events go to the dashboard.
        log.debug("call event %s", event.get("type"))


async def handle_twilio_stream(ws: WebSocket, settings: Settings) -> None:
    await ws.accept()
    transport = TwilioTransport(ws)
    session: CallSession | None = None
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")

            if event == "start":
                start = msg["start"]
                transport.stream_sid = start["streamSid"]
                session = CallSession(
                    transport, settings,
                    call_id=start.get("callSid", transport.stream_sid)[-12:],
                )
                await session.start()
                log.info("twilio call started sid=%s", transport.stream_sid)

            elif event == "media" and session is not None:
                # Inbound is mu-law 8 kHz; one stateful codec per call leg.
                await session.on_audio(
                    transport.codec.decode(unb64(msg["media"]["payload"]))
                )

            elif event == "stop":
                break

    except Exception:
        log.exception("twilio stream error")
    finally:
        if session is not None:
            await session.close()
            log.info("twilio call metrics %s", session.metrics.summary())

"""FastAPI app: browser demo, telephony webhook, metrics.

Routes
    GET  /                            the demo page
    WS   /ws/call                     browser call leg (PCM16 @ 16 kHz)
    POST /telephony/twilio/voice      TwiML -- points a number at the stream
    WS   /telephony/twilio/stream     Twilio Media Streams leg
    GET  /healthz                     liveness + which providers resolved
    GET  /metrics/calls               latency traces for recent calls
"""
from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .orchestrator import CallSession
from .providers import build_llm, build_stt, build_tts, describe
from .telephony.twilio import handle_twilio_stream, twiml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
log = logging.getLogger("voiceagent")

WEB_DIR = Path(__file__).resolve().parents[2] / "web"

app = FastAPI(title="voiceagent", version="0.1.0")

# Recent calls, for the dashboard. In production this is your metrics backend;
# a bounded deque is the right amount of infrastructure for a demo.
RECENT_CALLS: deque[dict[str, Any]] = deque(maxlen=25)

if WEB_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> Any:
    page = WEB_DIR / "index.html"
    if not page.is_file():
        return HTMLResponse("<h1>voiceagent</h1><p>web/index.html is missing.</p>")
    return FileResponse(page)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    s = get_settings()
    stt, llm, tts = build_stt(s), build_llm(s), build_tts(s)
    try:
        return JSONResponse({
            "ok": True,
            "configured": {
                "stt": s.stt_provider, "llm": s.llm_provider, "tts": s.tts_provider,
            },
            # What actually loaded. A configured provider that failed to start
            # shows up here as a mock -- the single most useful line in this
            # response when a deploy "works" but sounds wrong.
            "resolved": describe(stt, llm, tts),
            "turn_taking": {
                "end_of_turn_silence_ms": s.end_of_turn_silence_ms,
                "barge_in_speech_ms": s.barge_in_speech_ms,
            },
        })
    finally:
        for p in (stt, llm, tts):
            try:
                await p.close()
            except Exception:
                pass


@app.get("/metrics/calls")
async def metrics_calls() -> JSONResponse:
    return JSONResponse({"calls": list(RECENT_CALLS)})


class BrowserTransport:
    """Call leg for the demo page.

    Audio goes over the socket as raw binary PCM16 @ 16 kHz; app events go as
    JSON text. Keeping them on one socket means event order and audio order
    cannot disagree, which matters when the dashboard is drawing a timeline.
    """

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws

    async def send_audio(self, pcm16k: bytes) -> None:
        await self.ws.send_bytes(pcm16k)

    async def clear_audio(self) -> None:
        await self.ws.send_text(json.dumps({"type": "clear"}))

    async def send_event(self, event: dict[str, Any]) -> None:
        await self.ws.send_text(json.dumps(event))


@app.websocket("/ws/call")
async def ws_call(ws: WebSocket) -> None:
    await ws.accept()
    settings = get_settings()
    session = CallSession(BrowserTransport(ws), settings)
    log.info("browser call %s providers=%s", session.call_id, session.providers)
    try:
        await session.start()
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await session.on_audio(data)
            elif (text := msg.get("text")) is not None:
                payload = json.loads(text)
                if payload.get("type") == "hangup":
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("browser call %s error", session.call_id)
    finally:
        await session.close()
        RECENT_CALLS.appendleft(session.metrics.to_dict())
        log.info("browser call %s summary=%s",
                 session.call_id, session.metrics.summary())


@app.post("/telephony/twilio/voice")
async def twilio_voice(request: Request) -> Response:
    s = get_settings()
    url = s.public_ws_url
    if not url:
        # Fail loudly: a silent fallback here is a dead phone call that looks
        # fine in the logs.
        return Response(
            content="PUBLIC_WS_URL is not configured", status_code=503,
            media_type="text/plain",
        )
    return Response(content=twiml(url), media_type="application/xml")


@app.websocket("/telephony/twilio/stream")
async def twilio_stream(ws: WebSocket) -> None:
    await handle_twilio_stream(ws, get_settings())

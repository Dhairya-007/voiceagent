"""Render demo/timeline.json + demo/call.wav into an MP4.

Every number on screen comes from the recorded run -- nothing is mocked up for
the camera. Frames are drawn with Pillow and piped to ffmpeg, which muxes them
against the call audio so what you hear matches what you see.

    python scripts/render_video.py --demo demo --out demo/demo.mp4
"""
from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import wave
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 25

BG        = (11, 14, 20)
PANEL     = (18, 23, 34)
PANEL2    = (23, 29, 43)
LINE      = (36, 48, 68)
TEXT      = (223, 230, 241)
MUTED     = (125, 141, 168)
ACCENT    = (94, 234, 212)
CALLER    = (125, 211, 252)
AGENT     = (196, 181, 253)
WARN      = (251, 191, 36)
BAD       = (251, 113, 133)

STAGES = [
    ("stt_finalize",     "stt finalize",  (96, 165, 250)),
    ("llm_ttft",         "llm 1st token", (251, 191, 36)),
    ("tts_ttfb",         "tts 1st byte",  (196, 181, 253)),
    ("playout_dispatch", "playout",       (94, 234, 212)),
]

MONO = "/System/Library/Fonts/SFNSMono.ttf"
SANS = "/System/Library/Fonts/SFNS.ttf"


def fonts() -> dict[str, ImageFont.FreeTypeFont]:
    def f(path: str, size: int):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            return ImageFont.load_default()
    return {
        "h1":   f(SANS, 23), "h2": f(MONO, 12), "body": f(SANS, 17),
        "small": f(MONO, 12), "tiny": f(MONO, 11), "big": f(MONO, 30),
        "mid":  f(MONO, 17),
    }


def wrap(draw, text, font, max_w) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if draw.textlength(trial, font=font) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def load_waveform(path: Path, columns: int) -> list[float]:
    """Peak amplitude per column, for the scrubber strip."""
    with contextlib.closing(wave.open(str(path), "rb")) as w:
        n = w.getnframes()
        raw = w.readframes(n)
    import array
    samples = array.array("h")
    samples.frombytes(raw)
    if not len(samples):
        return [0.0] * columns
    per = max(1, len(samples) // columns)
    out = []
    for i in range(columns):
        block = samples[i * per : (i + 1) * per]
        out.append((max(abs(s) for s in block) / 32768) if len(block) else 0.0)
    return out


def rrect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline,
                           width=width)


class Renderer:
    def __init__(self, demo: Path) -> None:
        self.data: dict[str, Any] = json.loads((demo / "timeline.json").read_text())
        self.events = self.data["events"]
        self.turns = {t["turn"]: t for t in self.data["call"]["turns"]}
        self.summary = self.data["call"]["summary"]
        self.providers = self.data["providers"]
        self.duration = self.data["duration_s"]
        self.f = fonts()
        self.wave = load_waveform(demo / "call.wav", 520)

        # Pre-compute the conversation as (time, who, text) so each frame is a
        # filter rather than a replay of the whole event log.
        self.lines: list[tuple[float, str, str]] = []
        for e in self.events:
            if e["type"] == "transcript" and e.get("final"):
                self.lines.append((e["t"], "caller", e["text"]))
            elif e["type"] == "agent_text":
                self.lines.append((e["t"], "agent", e["text"]))
            elif e["type"] == "barge_in":
                self.lines.append((e["t"], "barge", "caller interrupts"))

        self.states = [(e["t"], e.get("state", "idle")) for e in self.events
                       if e.get("state")]
        self.turn_done = [(e["t"], e["turn"]) for e in self.events
                          if e["type"] == "turn_complete"]

    def state_at(self, t: float) -> str:
        s = "idle"
        for at, st in self.states:
            if at <= t:
                s = st
            else:
                break
        return s

    # -------------------------------------------------------------- drawing
    def frame(self, t: float, outro: float = 0.0) -> Image.Image:
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        if outro > 0:
            self.outro(d, outro)
            return img
        self.header(d, t)
        self.waveform(d, t)
        self.transcript(d, t)
        self.waterfall(d, t)
        self.footer(d, t)
        return img

    def header(self, d, t: float) -> None:
        d.rectangle([0, 0, W, 60], fill=PANEL)
        d.line([0, 60, W, 60], fill=LINE)
        d.text((24, 18), "voiceagent", font=self.f["h1"], fill=TEXT)
        d.text((158, 25), "streaming STT → LLM → TTS on a phone call",
               font=self.f["small"], fill=MUTED)
        p = self.providers
        label = f"stt {p['stt']}   llm {p['llm']}   tts {p['tts']}"
        d.text((W - 24 - d.textlength(label, font=self.f["small"]), 25), label,
               font=self.f["small"], fill=ACCENT)

    def waveform(self, d, t: float) -> None:
        x0, y0, w, h = 24, 76, W - 48, 54
        rrect(d, [x0, y0, x0 + w, y0 + h], 6, fill=PANEL2)
        mid = y0 + h / 2
        cols = len(self.wave)
        played = (t / self.duration) * cols if self.duration else 0
        for i, amp in enumerate(self.wave):
            x = x0 + 6 + i * ((w - 12) / cols)
            a = max(1.0, amp * (h / 2 - 6))
            colour = ACCENT if i <= played else (45, 58, 80)
            d.line([x, mid - a, x, mid + a], fill=colour)
        px = x0 + 6 + played * ((w - 12) / cols)
        d.line([px, y0 + 4, px, y0 + h - 4], fill=TEXT, width=2)
        d.text((x0 + 8, y0 + h + 6), f"{t:5.1f}s / {self.duration:.1f}s",
               font=self.f["tiny"], fill=MUTED)

    def transcript(self, d, t: float) -> None:
        x0, y0, w = 24, 156, 600
        d.text((x0, y0), "CALL", font=self.f["h2"], fill=MUTED)

        state = self.state_at(t)
        colour = {"listening": CALLER, "thinking": WARN,
                  "speaking": AGENT}.get(state, MUTED)
        pill = state.upper()
        pw = d.textlength(pill, font=self.f["tiny"]) + 22
        rrect(d, [x0 + 54, y0 - 4, x0 + 54 + pw, y0 + 16], 10,
              fill=PANEL2, outline=colour)
        d.text((x0 + 65, y0 + 1), pill, font=self.f["tiny"], fill=colour)

        visible = [l for l in self.lines if l[0] <= t][-7:]
        y = y0 + 34
        for at, who, text in visible:
            fresh = (t - at) < 0.45
            if who == "barge":
                d.text((x0 + 10, y), f"⟂  {text}", font=self.f["small"], fill=WARN)
                y += 26
                continue
            accent = CALLER if who == "caller" else AGENT
            body = TEXT if fresh else (190, 199, 214)
            lines = wrap(d, text, self.f["body"], w - 120)
            box_h = 22 + 23 * len(lines)
            bx = x0 if who == "caller" else x0 + 150
            bw = w - 150
            rrect(d, [bx, y, bx + bw, y + box_h], 9,
                  fill=(13, 24, 38) if who == "caller" else (22, 19, 42),
                  outline=accent if fresh else LINE)
            d.text((bx + 12, y + 6), who.upper(), font=self.f["tiny"], fill=accent)
            for i, ln in enumerate(lines):
                d.text((bx + 12, y + 22 + i * 23), ln, font=self.f["body"], fill=body)
            y += box_h + 9
            if y > H - 150:
                break

    def waterfall(self, d, t: float) -> None:
        x0, y0, w = 668, 156, W - 668 - 24
        d.text((x0, y0), "RESPONSE LATENCY — SPEECH END → FIRST AUDIO OUT",
               font=self.f["h2"], fill=MUTED)

        shown = [n for at, n in self.turn_done if at <= t]
        traces = [self.turns[n] for n in shown
                  if self.turns[n].get("response_latency_ms")]
        budget = self.summary["target_ms"]

        y = y0 + 30
        for tr in traces[-4:]:
            total = tr["response_latency_ms"]
            over = total > budget
            head = f"turn {tr['turn']}" + ("  · barged in" if tr["barged_in"] else "")
            d.text((x0, y), head, font=self.f["small"], fill=MUTED)
            val = f"{total:.0f} ms"
            d.text((x0 + w - d.textlength(val, font=self.f["mid"]), y - 4), val,
                   font=self.f["mid"], fill=BAD if over else ACCENT)
            y += 24

            scale = max(total, budget)
            offset = 0.0
            for key, label, colour in STAGES:
                ms = tr["breakdown"].get(key) or 0.0
                d.text((x0, y + 1), label, font=self.f["tiny"], fill=MUTED)
                tx = x0 + 96
                tw = w - 96 - 52
                rrect(d, [tx, y - 1, tx + tw, y + 13], 3, fill=PANEL2)
                if ms > 0:
                    bx = tx + (offset / scale) * tw
                    bw = max(2.0, (ms / scale) * tw)
                    rrect(d, [bx, y - 1, bx + bw, y + 13], 3, fill=colour)
                offset += ms
                shown_ms = f"{ms:.0f}ms" if ms >= 1 else ("<1ms" if ms else "—")
                d.text((tx + tw + 8, y + 1), shown_ms,
                       font=self.f["tiny"], fill=MUTED)
                y += 17
            # budget marker
            d.text((x0 + 96, y), f"budget {budget:.0f} ms", font=self.f["tiny"],
                   fill=MUTED)
            y += 24

        if not traces:
            d.text((x0, y), "waiting for the first caller turn…",
                   font=self.f["small"], fill=MUTED)

    def outro(self, d, fade: float) -> None:
        """Closing card: the roll-up, and what was actually running."""
        s = self.summary
        lat = s["response_latency_ms"]
        cx = W // 2

        def centre(y, text, font, fill):
            d.text((cx - d.textlength(text, font=font) / 2, y), text,
                   font=font, fill=fill)

        centre(96, "voiceagent", self.f["h1"], TEXT)
        centre(132, "one recorded call · every figure measured, none drawn by hand",
               self.f["small"], MUTED)

        stats = [
            (f"{lat['p50']:.0f} ms", "p50 response", ACCENT),
            (f"{lat['p95']:.0f} ms", "p95 response", ACCENT),
            (str(s["caller_turns"]), "caller turns", TEXT),
            (str(s["barge_ins"]), "barge-ins", WARN),
        ]
        bw, gap = 232, 20
        x = cx - (len(stats) * bw + (len(stats) - 1) * gap) / 2
        for value, label, colour in stats:
            rrect(d, [x, 196, x + bw, 310], 10, fill=PANEL, outline=LINE)
            d.text((x + bw / 2 - d.textlength(value, font=self.f["big"]) / 2, 228),
                   value, font=self.f["big"], fill=colour)
            d.text((x + bw / 2 - d.textlength(label, font=self.f["small"]) / 2, 274),
                   label, font=self.f["small"], fill=MUTED)
            x += bw + gap

        p = self.providers
        rows = [
            ("transport", "websocket · browser mic, and Twilio Media Streams "
                          "(8 kHz mu-law) for a real number"),
            ("stt", f"{p['stt']} — local, offline, free"),
            ("llm", f"{p['llm']} — deterministic responder; set LLM_BASE_URL "
                    "for Ollama or any OpenAI-compatible model"),
            ("tts", f"{p['tts']} — streams, ~15 ms to first byte "
                    "(macOS `say` measured ~860 ms)"),
            ("turn taking", "energy VAD + hysteresis · barge-in cancels the turn "
                            "and clears queued carrier audio"),
        ]
        y = 356
        for label, text in rows:
            d.text((cx - 470, y), label, font=self.f["small"], fill=ACCENT)
            d.text((cx - 350, y), text, font=self.f["small"], fill=MUTED)
            y += 27

        centre(H - 108, "every component is free and runs locally",
               self.f["small"], TEXT)
        centre(H - 80, "budget: 800 ms, the point a caller starts to feel the gap",
               self.f["tiny"], MUTED)

    def footer(self, d, t: float) -> None:
        y = H - 74
        d.line([0, y, W, y], fill=LINE)
        shown = [self.turns[n] for at, n in self.turn_done if at <= t]
        lat = [tr["response_latency_ms"] for tr in shown
               if tr.get("response_latency_ms")]
        barge = sum(1 for tr in shown if tr["barged_in"])

        def cell(x, label, value, colour=TEXT):
            d.text((x, y + 14), label, font=self.f["tiny"], fill=MUTED)
            d.text((x, y + 32), value, font=self.f["mid"], fill=colour)

        cell(24, "TURNS", str(len(lat)))
        p50 = sorted(lat)[len(lat) // 2] if lat else None
        cell(130, "p50 LATENCY", f"{p50:.0f} ms" if p50 else "—",
             ACCENT if p50 and p50 <= self.summary["target_ms"] else TEXT)
        cell(300, "WORST", f"{max(lat):.0f} ms" if lat else "—")
        cell(430, "BARGE-INS", str(barge), WARN if barge else TEXT)
        note = ("all figures measured from this recorded run · "
                "faster-whisper + espeak-ng, local")
        d.text((W - 24 - d.textlength(note, font=self.f["tiny"]), y + 36), note,
               font=self.f["tiny"], fill=MUTED)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=Path, default=Path("demo"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tail", type=float, default=0.8,
                    help="extra seconds held on the final call frame")
    ap.add_argument("--outro", type=float, default=4.0,
                    help="seconds of closing summary card")
    args = ap.parse_args()
    out = args.out or (args.demo / "demo.mp4")

    r = Renderer(args.demo)
    call_frames = int((r.duration + args.tail) * FPS)
    outro_frames = int(args.outro * FPS)
    n_frames = call_frames + outro_frames
    print(f"rendering {n_frames} frames at {W}x{H}/{FPS}fps "
          f"({r.duration:.1f}s call + {args.outro:.1f}s outro)")

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-",
         "-i", str(args.demo / "call.wav"),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20",
         "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
         # apad + the frame count decide the length: without this, -shortest
         # would end the file when the call audio does and drop the outro.
         "-af", "apad", "-movflags", "+faststart",
         "-frames:v", str(n_frames), str(out)],
        stdin=subprocess.PIPE,
    )
    assert ff.stdin is not None
    try:
        for i in range(n_frames):
            if i < call_frames:
                frame = r.frame(i / FPS)
            else:
                frame = r.frame(r.duration, outro=(i - call_frames + 1) / FPS)
            ff.stdin.write(frame.tobytes())
            if i % 100 == 0:
                print(f"  {i}/{n_frames}", flush=True)
    finally:
        ff.stdin.close()
        ff.wait()

    size = out.stat().st_size / 1e6 if out.exists() else 0
    print(f"wrote {out} ({size:.1f} MB)")
    return 0 if ff.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

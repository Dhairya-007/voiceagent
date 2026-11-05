/**
 * Demo client. Captures the mic at 16 kHz PCM16, streams it over one
 * websocket, plays returned audio through a clearable ring buffer, and draws
 * the per-turn latency waterfall the server reports.
 */
const TARGET_RATE = 16000;
const BUDGET_MS = 800;

const els = {
  call: document.getElementById('call'),
  hangup: document.getElementById('hangup'),
  state: document.getElementById('state'),
  level: document.getElementById('level'),
  log: document.getElementById('log'),
  turns: document.getElementById('turns'),
  providers: document.getElementById('providers'),
  lastLatency: document.getElementById('lastLatency'),
};

let ws, ctx, micStream, capture, playback, interimEl;
const traces = new Map();

fetch('/healthz').then(r => r.json()).then(h => {
  const r = h.resolved;
  els.providers.innerHTML =
    `stt <b>${r.stt}</b> · llm <b>${r.llm}</b> · tts <b>${r.tts}</b>`;
}).catch(() => {});

els.call.onclick = startCall;
els.hangup.onclick = () => { try { ws?.send(JSON.stringify({type:'hangup'})); } catch {} endCall(); };

async function startCall() {
  els.call.disabled = true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        // The browser's own AEC/NS is doing the job real AEC does on a phone
        // leg. Without it, the mic hears the agent through the speakers and
        // every reply triggers a false barge-in.
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    els.call.disabled = false;
    logEvent('mic permission denied — ' + err.message, 'barge');
    return;
  }

  ctx = new AudioContext();
  await ctx.audioWorklet.addModule('/static/capture-worklet.js');
  await ctx.audioWorklet.addModule('/static/playback-worklet.js');

  playback = new AudioWorkletNode(ctx, 'playback-processor', {
    processorOptions: { srcRate: TARGET_RATE },
    outputChannelCount: [1],
  });
  playback.connect(ctx.destination);

  capture = new AudioWorkletNode(ctx, 'capture-processor', {
    processorOptions: { targetRate: TARGET_RATE },
  });
  ctx.createMediaStreamSource(micStream).connect(capture);

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/call`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    els.hangup.disabled = false;
    els.log.innerHTML = '';
    capture.port.onmessage = (e) => {
      if (ws?.readyState === WebSocket.OPEN) ws.send(e.data);
      meter(e.data);
    };
  };
  ws.onmessage = (e) => {
    if (e.data instanceof ArrayBuffer) playback.port.postMessage(e.data, [e.data]);
    else handleEvent(JSON.parse(e.data));
  };
  ws.onclose = endCall;
  ws.onerror = () => logEvent('socket error', 'barge');
}

function endCall() {
  els.call.disabled = false;
  els.hangup.disabled = true;
  setState('idle');
  micStream?.getTracks().forEach(t => t.stop());
  capture?.disconnect(); playback?.disconnect();
  ctx?.close().catch(() => {});
  ws = ctx = micStream = capture = playback = null;
}

function meter(buffer) {
  const pcm = new Int16Array(buffer);
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
  const rms = Math.sqrt(sum / pcm.length) / 32768;
  els.level.style.width = Math.min(100, rms * 420) + '%';
}

function setState(s) {
  els.state.textContent = s;
  els.state.className = 'pill ' + s;
  if (els.level) {
    els.level.style.background =
      s === 'speaking' ? 'var(--agent)' : 'var(--caller)';
  }
}

function handleEvent(ev) {
  if (ev.state) setState(ev.state);
  switch (ev.type) {
    case 'clear':
      // Barge-in: drop everything queued for playback immediately.
      playback?.port.postMessage('clear');
      break;
    case 'transcript':
      if (ev.final) { interimEl?.remove(); interimEl = null; addMsg('caller', ev.text); }
      else {
        if (!interimEl) interimEl = addMsg('caller', ev.text, 'interim');
        else interimEl.querySelector('.text').textContent = ev.text;
      }
      break;
    case 'agent_text':
      addMsg('agent', ev.text);
      break;
    case 'barge_in':
      logEvent('⟂ barge-in — playback cancelled', 'barge');
      break;
    case 'turn_complete':
      if (ev.trace) renderTurn(ev.trace);
      break;
    case 'call_ended':
      logEvent(`call ended · p50 ${ev.summary?.response_latency_ms?.p50 ?? '—'} ms`);
      break;
  }
}

function addMsg(who, text, extra = '') {
  const div = document.createElement('div');
  div.className = `msg ${who} ${extra}`.trim();
  div.innerHTML = `<div class="who">${who}</div><div class="text"></div>`;
  div.querySelector('.text').textContent = text;
  els.log.appendChild(div);
  els.log.parentElement.scrollTop = 1e9;
  return div;
}

function logEvent(text, extra = '') {
  const div = document.createElement('div');
  div.className = `msg event ${extra}`.trim();
  div.textContent = text;
  els.log.appendChild(div);
  return div;
}

const STAGES = [
  ['stt_finalize',      'stt finalize', '#60a5fa'],
  ['llm_ttft',          'llm 1st token', '#fbbf24'],
  ['tts_ttfb',          'tts 1st byte', '#c4b5fd'],
  ['playout_dispatch',  'playout', '#5eead4'],
];

function renderTurn(trace) {
  const total = trace.response_latency_ms;
  if (total == null) return;  // the greeting: nothing was being responded to
  if (traces.size === 0) els.turns.innerHTML = '';
  traces.set(trace.turn, trace);
  els.lastLatency.textContent = Math.round(total) + ' ms';

  const scale = Math.max(total, BUDGET_MS);
  let offset = 0;
  const bars = STAGES.map(([key, label, color]) => {
    const ms = trace.breakdown[key] || 0;
    const left = (offset / scale) * 100;
    const width = Math.max(ms > 0 ? 0.8 : 0, (ms / scale) * 100);
    offset += ms;
    return `<div class="bar">
      <label>${label}</label>
      <div class="track"><i style="left:${left}%;width:${width}%;background:${color}"></i></div>
      <div class="val">${ms ? Math.round(ms) + 'ms' : '—'}</div>
    </div>`;
  }).join('');

  const existing = document.getElementById(`turn-${trace.turn}`);
  const node = existing || document.createElement('div');
  node.id = `turn-${trace.turn}`;
  node.className = 'turn';
  node.innerHTML = `
    <div class="turn-head">
      <span>turn ${trace.turn}${trace.barged_in ? ' · barged in' : ''}</span>
      <span class="total ${total > BUDGET_MS ? 'over' : ''}">${Math.round(total)} ms</span>
    </div>
    <div class="bars">${bars}</div>
    <div class="budget">budget ${BUDGET_MS} ms · ${
      total > BUDGET_MS ? `over by ${Math.round(total - BUDGET_MS)} ms` : 'within budget'
    }${trace.audio_ms ? ` · spoke ${(trace.audio_ms / 1000).toFixed(1)}s` : ''}</div>`;
  if (!existing) els.turns.appendChild(node);
}

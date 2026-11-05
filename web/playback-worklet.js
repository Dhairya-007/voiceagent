/**
 * Playback: a ring buffer of 16 kHz PCM16 resampled up to the device rate.
 *
 * Two requirements the obvious `new Audio()` per chunk cannot meet:
 *
 *  1. Gapless. Chunks arrive every ~100 ms; any seam between them is an
 *     audible click, and a hundred clicks a minute sounds broken.
 *  2. Instantly clearable. Barge-in has to drop queued audio *now*. Here that
 *     is one assignment -- the write head is reset and the buffer is empty.
 */
class PlaybackProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.srcRate = options.processorOptions.srcRate || 16000;
    this.ratio = this.srcRate / sampleRate;
    this.buf = new Float32Array(sampleRate * 20);
    this.write = 0;
    this.read = 0;
    this.port.onmessage = (e) => {
      if (e.data === 'clear') {
        this.read = this.write = 0;
        return;
      }
      const pcm = new Int16Array(e.data);
      for (let i = 0; i < pcm.length; i++) {
        this.buf[this.write % this.buf.length] = pcm[i] / 32768;
        this.write++;
      }
    };
  }

  process(_inputs, outputs) {
    const out = outputs[0][0];
    const available = this.write - this.read;
    if (available <= 1) {
      out.fill(0);
      return true;
    }
    for (let i = 0; i < out.length; i++) {
      const pos = this.read;
      const idx = Math.floor(pos);
      if (idx + 1 >= this.write) { out.fill(0, i); break; }
      const frac = pos - idx;
      const a = this.buf[idx % this.buf.length];
      const b = this.buf[(idx + 1) % this.buf.length];
      out[i] = a * (1 - frac) + b * frac;
      this.read += this.ratio;
    }
    return true;
  }
}
registerProcessor('playback-processor', PlaybackProcessor);

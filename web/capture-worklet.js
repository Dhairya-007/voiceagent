/**
 * Mic capture: AudioContext rate (usually 48 kHz float32) -> 16 kHz PCM16.
 *
 * Done in a worklet rather than on the main thread so a busy UI -- redrawing
 * the latency chart, say -- cannot drop audio frames. A dropped frame is a
 * clipped word, and a clipped word is a wrong transcript.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions.targetRate || 16000;
    this.ratio = sampleRate / this.targetRate;
    this.pos = 0;          // fractional read position in the input stream
    this.tail = new Float32Array(0);
    this.out = [];
    this.outFrames = Math.round(this.targetRate * 0.02); // 20 ms packets
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    // Prepend whatever was left over so interpolation spans the block seam.
    const block = new Float32Array(this.tail.length + input[0].length);
    block.set(this.tail, 0);
    block.set(input[0], this.tail.length);

    let p = this.pos;
    while (p + 1 < block.length) {
      const i = Math.floor(p);
      const frac = p - i;
      // Linear interpolation. Not a brick-wall anti-alias filter, but at
      // 48k -> 16k on speech the artefacts sit above the band STT cares about.
      const s = block[i] * (1 - frac) + block[i + 1] * frac;
      this.out.push(Math.max(-1, Math.min(1, s)));
      p += this.ratio;
    }
    const consumed = Math.floor(p);
    this.tail = block.slice(consumed);
    this.pos = p - consumed;

    while (this.out.length >= this.outFrames) {
      const slice = this.out.splice(0, this.outFrames);
      const pcm = new Int16Array(slice.length);
      for (let i = 0; i < slice.length; i++) {
        pcm[i] = slice[i] < 0 ? slice[i] * 0x8000 : slice[i] * 0x7fff;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }
    return true;
  }
}
registerProcessor('capture-processor', CaptureProcessor);

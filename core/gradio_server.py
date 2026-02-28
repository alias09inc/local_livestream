"""Gradio server for browser-to-WSL realtime transcription."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gradio as gr
import numpy as np

from .asr import SAMPLE_RATE, load_asr_model

logger = logging.getLogger(__name__)

SILENCE_THRESHOLD = 0.001
SILENCE_CHUNKS_NEEDED = 4
MIN_UTTERANCE_SAMPLES = int(0.5 * SAMPLE_RATE)
MAX_UTTERANCE_SAMPLES = int(12.0 * SAMPLE_RATE)


def _is_silent(chunk: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    if chunk.size == 0:
        return True
    rms = np.sqrt(np.mean(chunk**2))
    return bool(rms < threshold)


def _to_mono_float32(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if np.issubdtype(arr.dtype, np.integer):
        max_val = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / float(max_val)
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, -1.0, 1.0)


def _resample(
    audio: np.ndarray, src_rate: int, dst_rate: int = SAMPLE_RATE
) -> np.ndarray:
    if src_rate == dst_rate or audio.size == 0:
        return audio
    src_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    dst_len = int(audio.size * dst_rate / src_rate)
    dst_x = np.linspace(0.0, 1.0, num=max(1, dst_len), endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


@dataclass
class StreamState:
    buffer: np.ndarray
    silence_counter: int
    lines: list[str]


def _default_state() -> StreamState:
    return StreamState(
        buffer=np.array([], dtype=np.float32), silence_counter=0, lines=[]
    )


class BrowserRealtimeTranscriber:
    def __init__(self) -> None:
        self.asr_pipe = load_asr_model()

    def _transcribe(self, audio: np.ndarray) -> str:
        texts: list[str] = []
        for result in self.asr_pipe(audio):
            text = result.get("text", "").strip()
            if text:
                texts.append(text)
        return " ".join(texts).strip()

    def process_stream(self, audio_chunk, state: StreamState | None):
        if state is None:
            state = _default_state()
        if audio_chunk is None:
            return "\n".join(state.lines), state

        sample_rate, data = audio_chunk
        chunk = _to_mono_float32(data)
        chunk = _resample(chunk, int(sample_rate), SAMPLE_RATE)

        if chunk.size == 0:
            return "\n".join(state.lines), state

        state.buffer = np.concatenate([state.buffer, chunk], axis=0)
        state.silence_counter = state.silence_counter + 1 if _is_silent(chunk) else 0

        should_flush = (
            state.silence_counter >= SILENCE_CHUNKS_NEEDED
            and state.buffer.size >= MIN_UTTERANCE_SAMPLES
        ) or state.buffer.size >= MAX_UTTERANCE_SAMPLES

        if should_flush:
            text = self._transcribe(state.buffer)
            if text:
                state.lines.append(text)
            state.buffer = np.array([], dtype=np.float32)
            state.silence_counter = 0

        return "\n".join(state.lines[-50:]), state

    def flush(self, state: StreamState | None):
        if state is None:
            state = _default_state()
        if state.buffer.size >= MIN_UTTERANCE_SAMPLES:
            text = self._transcribe(state.buffer)
            if text:
                state.lines.append(text)
        state.buffer = np.array([], dtype=np.float32)
        state.silence_counter = 0
        return "\n".join(state.lines[-50:]), state

    def clear(self):
        state = _default_state()
        return "", state


def build_app() -> gr.Blocks:
    transcriber = BrowserRealtimeTranscriber()

    with gr.Blocks(title="Qwen3-ASR Realtime (WSL)") as app:
        gr.Markdown(
            "# Qwen3-ASR Realtime Transcription\n"
            "ブラウザのマイク音声をWSL上のASRサーバーへ送信して、リアルタイム文字起こしします。"
        )

        mic = gr.Audio(
            sources=["microphone"],
            type="numpy",
            streaming=True,
            label="Microphone (Browser)",
        )
        transcript = gr.Textbox(
            label="Transcript",
            lines=16,
            interactive=False,
            placeholder="発話するとここに追記されます",
        )

        state = gr.State(_default_state())
        flush_btn = gr.Button("Flush current buffer")
        clear_btn = gr.Button("Clear transcript")

        mic.stream(
            fn=transcriber.process_stream,
            inputs=[mic, state],
            outputs=[transcript, state],
            concurrency_limit=1,
        )
        flush_btn.click(
            fn=transcriber.flush,
            inputs=[state],
            outputs=[transcript, state],
        )
        clear_btn.click(
            fn=transcriber.clear,
            inputs=None,
            outputs=[transcript, state],
        )

    return app


def run_gradio_server(host: str = "0.0.0.0", port: int = 7860) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logger.info("Starting Gradio server at http://%s:%s", host, port)
    app = build_app()
    app.launch(server_name=host, server_port=port, inbrowser=False)

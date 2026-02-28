"""Microphone capture with silence-based utterance segmentation."""

from __future__ import annotations

import logging
import queue
import time
from collections import deque
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except OSError:
    sd = None
except ImportError:
    sd = None

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
CHUNK_DURATION_MS = 200
SILENCE_DURATION_MS = 900
MIN_UTTERANCE_MS = 500
MAX_UTTERANCE_MS = 12_000
SILENCE_THRESHOLD = 0.001


def is_silent(chunk: np.ndarray, threshold: float = SILENCE_THRESHOLD) -> bool:
    if chunk.size == 0:
        return True
    rms = np.sqrt(np.mean(chunk**2))
    return bool(rms < threshold)


class AudioCapture:
    """Continuously captures microphone audio and emits utterances via queue."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration_ms: int = CHUNK_DURATION_MS,
        silence_duration_ms: int = SILENCE_DURATION_MS,
        min_utterance_ms: int = MIN_UTTERANCE_MS,
        max_utterance_ms: int = MAX_UTTERANCE_MS,
        silence_threshold: float = SILENCE_THRESHOLD,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.silence_threshold = silence_threshold

        self.frames_per_chunk = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_chunks_needed = max(
            1, int(silence_duration_ms / chunk_duration_ms)
        )
        self.min_utterance_samples = int(sample_rate * min_utterance_ms / 1000)
        self.max_utterance_samples = int(sample_rate * max_utterance_ms / 1000)

        self.audio_buffer: deque[np.ndarray] = deque()
        self.audio_queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self.running = True

    def _audio_callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            logger.info("Input stream status: %s", status)
        self.audio_buffer.append(indata[:, 0].copy())

    def recorder_loop(self) -> None:
        """Capture loop: segments utterances based on trailing silence."""
        if sd is None:
            raise RuntimeError(
                "sounddevice/PortAudio is not available. "
                "Use the Gradio server mode on WSL."
            )

        silence_counter = 0
        logger.info("Opening microphone stream (sample_rate=%s)", self.sample_rate)

        with sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            dtype="float32",
            blocksize=self.frames_per_chunk,
            latency="low",
            callback=self._audio_callback,
        ):
            while self.running:
                time.sleep(self.chunk_duration_ms / 1000.0)
                if not self.audio_buffer:
                    continue

                latest = self.audio_buffer[-1]
                if is_silent(latest, self.silence_threshold):
                    silence_counter += 1
                else:
                    silence_counter = 0

                buffer_samples = sum(chunk.size for chunk in self.audio_buffer)
                is_max_reached = buffer_samples >= self.max_utterance_samples
                is_end_of_speech = silence_counter >= self.silence_chunks_needed

                if not is_max_reached and not is_end_of_speech:
                    continue

                utterance = np.concatenate(list(self.audio_buffer), axis=0)
                self.audio_buffer.clear()
                silence_counter = 0

                if utterance.size < self.min_utterance_samples:
                    continue

                self.audio_queue.put(utterance)
                logger.debug(
                    "Queued utterance %.2fs", utterance.size / self.sample_rate
                )

    def stop(self) -> None:
        self.running = False
        self.audio_queue.put(None)

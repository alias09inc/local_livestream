"""Run realtime transcription with Qwen3-ASR."""

from __future__ import annotations

import logging
import threading
import time

from .asr import load_asr_model, stream_generator
from .audio import AudioCapture


class RealtimeTranscriber:
    def __init__(self) -> None:
        self.audio_capture = AudioCapture()
        self.asr_pipe = None

    def _recording_thread(self) -> None:
        self.audio_capture.recorder_loop()

    def _transcription_thread(self) -> None:
        self.asr_pipe = load_asr_model()
        for result in self.asr_pipe(
            stream_generator(self.audio_capture.audio_queue),
            batch_size=1,
            generate_kwargs={"max_new_tokens": 256},
        ):
            text = result.get("text", "").strip()
            if text:
                print(text, flush=True)

    def run(self) -> None:
        rec_thread = threading.Thread(target=self._recording_thread, daemon=True)
        asr_thread = threading.Thread(target=self._transcription_thread, daemon=True)
        rec_thread.start()
        asr_thread.start()

        logging.info("Realtime transcription started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info("Stopping transcription...")
            self.audio_capture.stop()
            rec_thread.join(timeout=3)
            asr_thread.join(timeout=3)


def run_realtime_transcription() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    RealtimeTranscriber().run()

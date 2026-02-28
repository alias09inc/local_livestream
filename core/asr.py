"""Qwen3-ASR realtime transcription helpers."""

from __future__ import annotations

import logging
from collections.abc import Generator
from typing import Optional, Union

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

logger = logging.getLogger(__name__)

ASR_MODEL_NAME = "Qwen/Qwen3-ASR-0.6B"
SAMPLE_RATE = 16_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


class QwenASRStreamWrapper:
    """Wraps Qwen3ASRModel to yield dicts similar to ASR pipelines."""

    def __init__(self, model: Qwen3ASRModel):
        self.model = model

    def __call__(
        self,
        audio_input: Union[np.ndarray, Generator[np.ndarray, None, None]],
        batch_size: int = 1,
        generate_kwargs: Optional[dict] = None,
        **_kwargs,
    ):
        if generate_kwargs:
            logger.debug("generate_kwargs currently unused: %s", generate_kwargs)
        if batch_size != 1:
            logger.debug("batch_size=%s ignored for streaming mode", batch_size)

        if isinstance(audio_input, np.ndarray):
            yield from self._transcribe_one(audio_input)
            return

        for chunk in audio_input:
            if chunk is None:
                break
            yield from self._transcribe_one(chunk)

    def _transcribe_one(self, audio: np.ndarray):
        audio_tuple = (np.asarray(audio, dtype=np.float32), SAMPLE_RATE)
        results = self.model.transcribe(audio=[audio_tuple], return_time_stamps=False)

        if isinstance(results, list):
            for item in results:
                yield {"text": getattr(item, "text", str(item))}
        else:
            yield {"text": getattr(results, "text", str(results))}


def load_asr_model(model_name: str = ASR_MODEL_NAME) -> QwenASRStreamWrapper:
    logger.info("Loading ASR model %s on %s", model_name, DEVICE)

    model = Qwen3ASRModel.from_pretrained(
        model_name,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    )
    return QwenASRStreamWrapper(model)


def stream_generator(audio_queue) -> Generator[np.ndarray, None, None]:
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

"""
Turning recorded speech into text with faster-whisper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from krrood.patterns.classproperty import classproperty

from cramspeech.microphone import Recording


# %% Transcript
@dataclass(kw_only=True)
class Transcript:
    """
    What a :class:`WhisperTranscriber` understood in a recording.
    """

    text: str
    """
    The recognized words, empty when nothing was said.
    """


# %% Whisper
class WhisperModelSize(StrEnum):
    """
    Whisper models as faster-whisper names them. Larger models are more accurate and slower.
    """

    TINY = "tiny"
    """
    Smallest multilingual model.
    """

    BASE = "base"
    """
    Second smallest multilingual model.
    """

    SMALL = "small"
    """
    Mid-sized multilingual model.
    """


class WhisperLanguage(StrEnum):
    """
    The languages a Whisper model can be told to expect.
    """

    ENGLISH = "en"
    """
    English speech.
    """

    GERMAN = "de"
    """
    German speech.
    """


class WhisperComputeType(StrEnum):
    """
    Number formats faster-whisper can run a Whisper model in. Smaller formats run faster
    and use less memory, at a small cost in accuracy.
    """

    INT8 = "int8"
    """
    8-bit integer weights, the fastest format on a CPU.
    """

    FLOAT16 = "float16"
    """
    16-bit floating point, the usual format on a GPU.
    """

    FLOAT32 = "float32"
    """
    32-bit floating point, the most accurate and slowest format.
    """


@dataclass(kw_only=True)
class WhisperTranscriber:
    """
    Transcribes speech with faster-whisper.
    """

    model_size: WhisperModelSize = WhisperModelSize.SMALL
    """
    Which Whisper model to run.
    """

    language: WhisperLanguage = WhisperLanguage.ENGLISH
    """
    The spoken language.
    """

    beam_size: int = 8
    """
    Number of candidate transcriptions kept during decoding.
    """

    compute_type: WhisperComputeType = WhisperComputeType.INT8
    """
    Number format the model runs in.
    """

    hotwords: list[str] = field(default_factory=list)
    """
    Words and phrases the speech is likely to hold, such as names and objects, so that
    Whisper prefers them over similar sounding words.

    .. note::
        faster-whisper keeps only the first 223 tokens of the hotwords, half of
        Whisper's context of 448 tokens, and drops the rest without warning. With its
        comma an English word takes about 2 tokens and a German compound up to 6, so
        about 90 English or 40 German hotwords fit. List the most important ones first.
    """

    @cached_property
    def model(self) -> WhisperModel:
        """
        The loaded Whisper model. faster-whisper downloads it from Hugging Face on first use.
        """
        return WhisperModel(self.model_size, compute_type=self.compute_type)

    @classproperty
    def sample_rate(cls) -> int:
        """
        :return: Samples per second, in hertz, that Whisper expects its recordings at.
        """
        return FeatureExtractor().sampling_rate

    def transcribe(self, *, recording: Recording) -> Transcript:
        """
        :param recording: Speech recorded at :attr:`sample_rate`.
        :return: The text spoken in the recording, empty when nobody spoke.
        :raises SampleRateMismatch: When the recording is not at :attr:`sample_rate`.
        """
        recording.require_sample_rate(model_sample_rate=self.sample_rate)
        segments, _ = self.model.transcribe(
            recording.waveform,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,
            hotwords=", ".join(self.hotwords),
        )
        return Transcript(text=" ".join(segment.text.strip() for segment in segments))

"""
Finding where somebody spoke and then paused in streamed audio, with the Silero voice
activity model that ships with faster-whisper.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from math import ceil

import numpy as np
from faster_whisper.vad import SileroVADModel, get_vad_model
from krrood.patterns.classproperty import classproperty

from cramspeech.microphone import Recording


# %% Speech detection
@dataclass(kw_only=True)
class UtteranceSearch:
    """
    What a :class:`SpeechDetector` found when it searched audio for an utterance.
    """

    utterance: Recording | None
    """
    The first finished utterance with its lead-in, ``None`` while nobody spoke or the
    speaker is still talking.
    """

    holds_speech: bool
    """
    Whether the searched audio holds any speech.
    """

    first_sample_to_keep: int
    """
    Index of the first sample of the searched audio that later searches still need.
    """


@dataclass(kw_only=True)
class SpeechDetector:
    """
    Finds where somebody spoke and then paused, using the Silero voice activity model
    that ships with faster-whisper.
    """

    speech_threshold: float = 0.5
    """
    Score from which the model counts a frame as speech, between 0 and 1.
    """

    pause_duration: float = 1.0
    """
    Seconds without speech after which an utterance is finished.
    """

    kept_before_speech: float = 0.4
    """
    Seconds of audio kept before the first speech frame, so the first word is not cut
    off.
    """

    maximum_duration: float = 30.0
    """
    Longest utterance, in seconds.

    Speech that goes on longer is cut off there.
    """

    @cached_property
    def model(self) -> SileroVADModel:
        """
        The Silero model that scores how likely each frame holds speech.
        """
        return get_vad_model()

    @classproperty
    def sample_rate(cls) -> int:
        """
        :return: Samples per second, in hertz, that Silero expects its recordings at.
        """
        return 16000

    @classproperty
    def frame_length(cls) -> int:
        """
        :return: Number of samples the model scores at once.
        """
        return 512

    def search_for_utterance(self, *, recording: Recording) -> UtteranceSearch:
        """
        :param recording: Audio at :attr:`sample_rate`, oldest sample first.
        :return: The first utterance that ended in a pause of :attr:`pause_duration` or
            reached :attr:`maximum_duration`, with :attr:`kept_before_speech` of audio
            before it, and the part of the audio later searches still need.
        :raises SampleRateMismatch: When the recording is not at :attr:`sample_rate`.
        """
        speech_frames = self.speech_frames_in(recording=recording)
        frame_count = len(recording.waveform) // self.frame_length
        lead_in = round(self.kept_before_speech * self.sample_rate)
        if speech_frames.size == 0:
            frames_before_lead_in = (len(recording.waveform) - lead_in) // (
                self.frame_length
            )
            return UtteranceSearch(
                utterance=None,
                holds_speech=False,
                first_sample_to_keep=max(0, frames_before_lead_in * self.frame_length),
            )

        start = max(0, int(speech_frames[0]) * self.frame_length - lead_in)
        maximum_end = start + round(self.maximum_duration * self.sample_rate)
        pause_start = self.start_of_first_pause(
            speech_frames=speech_frames, frame_count=frame_count
        )
        if pause_start is None and len(recording.waveform) < maximum_end:
            return UtteranceSearch(
                utterance=None, holds_speech=True, first_sample_to_keep=0
            )
        end = maximum_end if pause_start is None else min(pause_start, maximum_end)
        utterance = Recording(
            waveform=recording.waveform[start:end], sample_rate=recording.sample_rate
        )
        return UtteranceSearch(
            utterance=utterance, holds_speech=True, first_sample_to_keep=end
        )

    def speech_frames_in(self, *, recording: Recording) -> np.ndarray:
        """
        :param recording: Audio at :attr:`sample_rate`, oldest sample first.
        :return: Indices of the whole frames that hold speech, in ascending order.
        :raises SampleRateMismatch: When the recording is not at :attr:`sample_rate`.
        """
        recording.require_sample_rate(model_sample_rate=self.sample_rate)
        frame_count = len(recording.waveform) // self.frame_length
        if frame_count == 0:
            return np.zeros(0, dtype=np.int64)
        scores = self.model(recording.waveform[: frame_count * self.frame_length])
        return np.flatnonzero(scores >= self.speech_threshold)

    def start_of_first_pause(
        self, *, speech_frames: np.ndarray, frame_count: int
    ) -> int | None:
        """
        :param speech_frames: Indices of the frames that hold speech, in ascending order.
        :param frame_count: Number of frames scored.
        :return: The sample at which the first silence of at least
            :attr:`pause_duration` begins, ``None`` if the speaker has not paused that
            long yet.
        """
        pause_frame_count = ceil(
            self.pause_duration * self.sample_rate / self.frame_length
        )
        frames_to_next_speech = np.diff(speech_frames, append=frame_count)
        paused_after = np.flatnonzero(frames_to_next_speech > pause_frame_count)
        if paused_after.size == 0:
            return None
        return (int(speech_frames[paused_after[0]]) + 1) * self.frame_length

"""
Hearing speech from a microphone and transcribing it, for a fixed time, one utterance at
a time, or in the background while a plan goes on.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from math import ceil
from queue import Queue
from threading import Event, Thread

import numpy as np
from typing_extensions import Iterator

from cramspeech.exceptions import NothingHeard
from cramspeech.microphone import Microphone, Recording
from cramspeech.speech_detection import SpeechDetector, UtteranceSearch
from cramspeech.transcription import Transcript, WhisperTranscriber


# %% Speech recognizer
@dataclass(kw_only=True)
class SpeechRecognizer:
    """
    Hears speech from a microphone and transcribes it, either for a fixed time or one
    utterance at a time.
    """

    microphone: Microphone = field(default_factory=Microphone)
    """
    The device the speech is recorded from.
    """

    transcriber: WhisperTranscriber = field(default_factory=WhisperTranscriber)
    """
    The speech-to-text model that turns the recording into text.
    """

    detector: SpeechDetector = field(default_factory=SpeechDetector)
    """
    Finds where an utterance in the streamed audio begins and ends.
    """

    block_duration: float = 0.25
    """
    Seconds of audio streamed from the microphone between two searches for a finished
    utterance.
    """

    @contextmanager
    def pause(self) -> Iterator[None]:
        """
        Stop hearing while the ``with`` statement runs, for example while the robot
        speaks.

        Audio arriving meanwhile is dropped, with any utterance under way when the pause
        began, also when it is read only after the pause.
        """
        self.microphone.is_muted.set()
        try:
            yield
        finally:
            self.microphone.is_muted.clear()

    def hear(self, *, duration: float) -> Transcript:
        """
        Record for a fixed time, starting now, and return what was said.

        :param duration: How long to record, in seconds.
        :return: The text spoken during the recording.
        """
        recording = self.microphone.record(
            duration=duration, sample_rate=self.transcriber.sample_rate
        )
        return self.transcriber.transcribe(recording=recording)

    def hear_next_utterance(self, *, timeout: float) -> Transcript:
        """
        Listen until somebody has spoken and paused, and return what was said.

        :param timeout: Seconds to wait for speech to start. Speech that has started is
            heard to its end.
        :return: The text of the first utterance.
        :raises NothingHeard: When no speech starts within *timeout*.
        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        timeout_block_count = ceil(timeout / self.block_duration)
        with closing(self._listen()) as searches:
            for block_count, search in enumerate(searches, start=1):
                if search.utterance is not None:
                    return self.transcriber.transcribe(recording=search.utterance)
                if block_count >= timeout_block_count and not search.holds_speech:
                    raise NothingHeard(timeout=timeout)

    def hear_utterances(
        self, *, stop_requested: Event | None = None
    ) -> Iterator[Transcript]:
        """
        Listen until *stop_requested* is set or the caller stops iterating, which
        releases the microphone.

        :param stop_requested: Ends listening at the next block once set, also while
            nobody speaks, and after the transcription under way, if any.
        :return: The text of every utterance with words in it, in spoken order, each as
            soon as its pause has been heard.
        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        with closing(self._listen()) as searches:
            for search in searches:
                if stop_requested is not None and stop_requested.is_set():
                    return
                if search.utterance is None:
                    continue
                transcript = self.transcriber.transcribe(recording=search.utterance)
                if transcript.text:
                    yield transcript

    def _listen(self) -> Iterator[UtteranceSearch]:
        """
        Listen to the microphone until the caller stops iterating, which releases it.

        :return: After each block of :attr:`block_duration`, the search of the audio
            heard so far, without the utterances already found. A block that arrived
            during :meth:`pause` yields an empty search.
        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        sample_rate = self.transcriber.sample_rate
        kept_audio = Recording(
            waveform=np.zeros(0, dtype=np.float32), sample_rate=sample_rate
        )
        with closing(
            self.microphone.stream(
                sample_rate=sample_rate, block_duration=self.block_duration
            )
        ) as blocks:
            for block in blocks:
                if block.size == 0:
                    kept_audio.waveform = kept_audio.waveform[:0]
                    yield UtteranceSearch(
                        utterance=None, holds_speech=False, first_sample_to_keep=0
                    )
                    continue
                kept_audio.waveform = np.concatenate((kept_audio.waveform, block))
                search = self.detector.search_for_utterance(recording=kept_audio)
                kept_audio.waveform = kept_audio.waveform[search.first_sample_to_keep :]
                yield search


# %% Listening in the background
@dataclass(kw_only=True)
class BackgroundListener:
    """
    Hears utterances in a thread of its own while the caller goes on, and keeps the
    sentences in a queue until the caller reads them.
    """

    recognizer: SpeechRecognizer
    """
    Hears and transcribes the utterances.
    """

    heard: Queue[Transcript] = field(default_factory=Queue, init=False)
    """
    Every sentence heard since :meth:`start`, in spoken order, until it is read.
    """

    stop_requested: Event = field(default_factory=Event, init=False)
    """
    Set by :meth:`stop` to end the listening.
    """

    thread: Thread | None = field(default=None, init=False)
    """
    The thread that listens, ``None`` before :meth:`start`.
    """

    def start(self) -> None:
        """
        Begin listening in a new thread and return at once.

        The thread does not keep the process alive, so a listener nobody stopped ends
        with the program.
        """
        self.stop_requested.clear()
        self.thread = Thread(target=self.queue_heard_sentences, daemon=True)
        self.thread.start()

    def queue_heard_sentences(self) -> None:
        """
        Put every sentence the recognizer hears into :attr:`heard` until
        :attr:`stop_requested` is set.

        :raises MicrophoneStopped: When the microphone stops delivering audio.
        """
        for transcript in self.recognizer.hear_utterances(
            stop_requested=self.stop_requested
        ):
            self.heard.put(transcript)

    def stop(self) -> None:
        """
        Stop listening at the next block, after the transcription under way if any, and
        wait until the microphone is released.
        """
        self.stop_requested.set()
        self.thread.join()

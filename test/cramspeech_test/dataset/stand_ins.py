"""
Stand-ins for the microphone driver, the loudspeaker, the ``piper`` program, the Whisper
model and the Silero model, so tests run without audio hardware, Piper or a model
download.
"""

from __future__ import annotations

# %% Imports
import subprocess
from dataclasses import dataclass, field
from itertools import chain, repeat
from threading import Event, Thread

import numpy as np
import sounddevice
from cramspeech.microphone import Recording
from cramspeech.speech_detection import SpeechDetector
from faster_whisper.transcribe import Segment
from krrood.patterns.called_positionally import called_positionally
from typing_extensions import Any, Callable, Iterator, Self


# %% Microphone driver
@dataclass(kw_only=True)
class ReplayingInputDevice:
    """
    Takes the place of :class:`sounddevice.InputStream`: as soon as the stream opens,
    plays a whole recording into its callback, as if the caller had been busy all along,
    then reports the stream finished.
    """

    recording: Recording
    """
    The audio the stream delivers, from its first sample on.
    """

    requests: list[dict[str, Any]] = field(default_factory=list)
    """
    The keyword arguments of each opened stream, its callbacks left out, in opening
    order.
    """

    is_open: bool = False
    """
    Whether a stream is currently open.
    """

    callback: (
        Callable[[np.ndarray, int, Any, sounddevice.CallbackFlags], None] | None
    ) = None
    """
    The function the latest stream hands each block of samples to.
    """

    finished_callback: Callable[[], None] | None = None
    """
    The function the latest stream calls once it delivers no more audio.
    """

    drops_audio: bool = False
    """
    Whether every block reports that audio before it was lost, as when the device's
    buffer overflowed.
    """

    def input_stream(
        self,
        *,
        callback: Callable[[np.ndarray, int, Any, sounddevice.CallbackFlags], None],
        finished_callback: Callable[[], None],
        **arguments: Any,
    ) -> ReplayingInputDevice:
        """
        :return: This device as the stream, keeping its callbacks and the keyword
            arguments it was opened with.
        """
        self.requests.append(arguments)
        self.callback = callback
        self.finished_callback = finished_callback
        return self

    def __enter__(self) -> Self:
        """
        Opens the stream and delivers the recording in blocks of the requested size,
        silence filling up the last one.

        Like PortAudio, every block reuses one buffer.
        """
        self.is_open = True
        blocks = self.recorded_blocks()
        block_length = blocks.shape[1]
        buffer = np.zeros((block_length, 1), dtype=np.float32)
        status = sounddevice.CallbackFlags()
        status.input_overflow = self.drops_audio
        for block in blocks:
            buffer[:, 0] = block
            self.callback(buffer, block_length, None, status)
        self.finished_callback()
        return self

    def recorded_blocks(self) -> np.ndarray:
        """
        :return: The recording as one row per block of the latest stream's size, silence
            filling up the last one.
        """
        block_length = self.requests[-1]["blocksize"]
        waveform = self.recording.waveform
        padded = np.pad(waveform, (0, -len(waveform) % block_length))
        return padded.reshape(-1, block_length)

    def __exit__(self, *exception_information: object) -> None:
        """
        Closes the stream, also when the caller raised.
        """
        self.is_open = False


@dataclass(kw_only=True)
class LiveInputDevice(ReplayingInputDevice):
    """
    Takes the place of :class:`sounddevice.InputStream` like a microphone in a quiet
    room:

    from its own thread, plays the recording into the stream's callback and then
    silence, block after block, until the stream is closed.
    """

    block_interval: float = 0.001
    """
    Seconds between two blocks, far shorter than a block lasts.
    """

    closed: Event = field(default_factory=Event)
    """
    Set once the stream is closed, which ends the delivery of blocks.
    """

    delivery: Thread | None = None
    """
    The thread that hands the blocks to the callback while the stream is open.
    """

    def __enter__(self) -> Self:
        """
        Opens the stream and starts delivering blocks of the requested size.
        """
        self.is_open = True
        self.closed.clear()
        self.delivery = Thread(target=self.deliver_blocks, daemon=True)
        self.delivery.start()
        return self

    def deliver_blocks(self) -> None:
        """
        Hand the recording to the callback, silence filling up its last block, and then
        silent blocks until the stream is closed.
        """
        recorded_blocks = self.recorded_blocks()
        block_length = recorded_blocks.shape[1]
        silence = np.zeros(block_length, dtype=np.float32)
        blocks = chain(recorded_blocks, repeat(silence))
        status = sounddevice.CallbackFlags()
        for block in blocks:
            if self.closed.wait(timeout=self.block_interval):
                return
            self.callback(block[:, np.newaxis], block_length, None, status)

    def __exit__(self, *exception_information: object) -> None:
        """
        Closes the stream once the delivering thread has stopped.
        """
        self.closed.set()
        self.delivery.join()
        self.is_open = False


@dataclass(kw_only=True)
class SteppedInputDevice(ReplayingInputDevice):
    """
    Takes the place of :class:`sounddevice.InputStream` like a microphone whose blocks
    arrive when the test says so: opening the stream delivers the first block of the
    recording, and each :meth:`deliver_next_block` the one after.
    """

    delivered_block_count: int = 0
    """
    Number of blocks of the recording handed to the callback so far.
    """

    def __enter__(self) -> Self:
        """
        Opens the stream and delivers only the first block.
        """
        self.is_open = True
        self.deliver_next_block()
        return self

    def deliver_next_block(self) -> None:
        """
        Hand the next block of the recording to the latest stream's callback.
        """
        block = self.recorded_blocks()[self.delivered_block_count]
        self.delivered_block_count += 1
        self.callback(
            block[:, np.newaxis], len(block), None, sounddevice.CallbackFlags()
        )


@dataclass(kw_only=True)
class AudioDevicesOfADesktop:
    """
    Takes the place of :func:`sounddevice.query_devices`: a desktop's devices, of which
    some only play, in the shape sounddevice lists them.
    """

    devices: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {"index": 0, "name": "HDMI output", "max_input_channels": 0},
            {"index": 1, "name": "USB headset", "max_input_channels": 1},
            {"index": 2, "name": "Built-in microphone", "max_input_channels": 2},
        ]
    )
    """
    Every device with its index, name and number of channels it records.
    """

    default_input_index: int = 2
    """
    The index of the system's default input.
    """

    def query_devices(self) -> list[dict[str, Any]]:
        """
        :return: :attr:`devices`.
        """
        return self.devices


@dataclass(kw_only=True)
class DefaultDevicesOfADesktop:
    """
    Takes the place of :data:`sounddevice.default`, holding only the default devices.
    """

    device: tuple[int, int]
    """
    The indices of the default input and output device.
    """


# %% Piper and the loudspeaker
@dataclass(kw_only=True)
class PiperProgram:
    """
    Takes the place of :func:`subprocess.run` for the ``piper`` program: answers every
    sentence with the same raw 16-bit audio and keeps every call.
    """

    samples: np.ndarray = field(
        default_factory=lambda: np.array([0, 1000, -1000, 32767], dtype=np.int16)
    )
    """
    The audio every sentence is spoken as.
    """

    calls: list[tuple[list[str], bytes]] = field(default_factory=list)
    """
    The command and the standard input of each call, in call order.
    """

    @called_positionally
    def run(self, command: list[str], **settings: Any) -> subprocess.CompletedProcess:
        """
        :return: A finished process that wrote :attr:`samples` to its standard output.
        """
        self.calls.append((command, settings["input"]))
        return subprocess.CompletedProcess(
            args=command, returncode=0, stdout=self.samples.tobytes(), stderr=b""
        )


@dataclass(kw_only=True)
class LoudspeakerKeepingWhatItPlays:
    """
    Takes the place of :func:`sounddevice.play` and :func:`sounddevice.wait`: plays
    nothing aloud and keeps every played audio with its sample rate.
    """

    played: list[tuple[np.ndarray, int]] = field(default_factory=list)
    """
    The samples and the sample rate of each :meth:`play` call, in call order.
    """

    waited: bool = False
    """
    Whether :meth:`wait` was called after the latest :meth:`play`.
    """

    @called_positionally
    def play(self, data: np.ndarray, samplerate: int) -> None:
        """
        Keep the audio instead of playing it.
        """
        self.played.append((data, samplerate))
        self.waited = False

    def wait(self) -> None:
        """
        Returns at once, since nothing is playing.
        """
        self.waited = True


# %% Whisper model
@dataclass(kw_only=True)
class ModelReturningSegments:
    """
    Takes the place of :class:`faster_whisper.WhisperModel`: hears the same texts in
    every recording and keeps the arguments of each transcription.
    """

    texts: list[str]
    """
    The text of each segment, in spoken order.
    """

    calls: list[tuple[np.ndarray, dict[str, Any]]] = field(default_factory=list)
    """
    The audio and keyword arguments of each :meth:`transcribe` call, in call order.
    """

    @called_positionally
    def transcribe(
        self, audio: np.ndarray, **settings: Any
    ) -> tuple[Iterator[Segment], None]:
        """
        :return: One :class:`~faster_whisper.transcribe.Segment` per text, lazily as Whisper
            yields them, and no transcription info.
        """
        self.calls.append((audio, settings))
        segments = (
            Segment(
                id=index,
                seek=0,
                start=0.0,
                end=0.0,
                text=text,
                tokens=[],
                avg_logprob=0.0,
                compression_ratio=0.0,
                no_speech_prob=0.0,
                words=None,
                temperature=None,
            )
            for index, text in enumerate(self.texts)
        )
        return segments, None


# %% Silero model
@dataclass(kw_only=True)
class ModelHearingSpeechInEverySound:
    """
    Takes the place of :class:`faster_whisper.vad.SileroVADModel`: every frame that
    holds any sound scores as certain speech, and every silent frame as certainly none.
    """

    scored_sample_counts: list[int] = field(default_factory=list)
    """
    The number of samples of each scored audio, in call order.
    """

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        :param audio: Whole frames of :attr:`SpeechDetector.frame_length` samples.
        :return: One score per frame, 1 for sound and 0 for silence.
        """
        self.scored_sample_counts.append(len(audio))
        frames = audio.reshape(-1, SpeechDetector.frame_length)
        return np.any(frames != 0, axis=1).astype(np.float32)

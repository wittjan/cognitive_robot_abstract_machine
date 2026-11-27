"""
Audio input: finding the input devices, and recording or streaming mono audio from one
of them.
"""

from __future__ import annotations

import logging
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from queue import Queue
from threading import Lock

import numpy as np
import sounddevice
from krrood.patterns.classproperty import classproperty
from typing_extensions import Any, Iterator

from cramspeech.exceptions import MicrophoneStopped, SampleRateMismatch


# %% Recording
@dataclass(kw_only=True)
class Recording:
    """
    Mono audio as a sequence of samples taken at a fixed rate.
    """

    waveform: np.ndarray
    """
    One float32 sample in [-1, 1] per frame, in recording order.
    """

    sample_rate: int
    """
    Samples per second, in hertz.
    """

    @classproperty
    def full_scale(cls) -> float:
        """
        :return: The largest magnitude a sample can have.
        """
        return 1.0

    def require_sample_rate(self, *, model_sample_rate: int) -> None:
        """
        :param model_sample_rate: Samples per second, in hertz, the model this recording
            is handed to takes.
        :raises SampleRateMismatch: When the recording is at another sample rate.
        """
        if self.sample_rate != model_sample_rate:
            raise SampleRateMismatch(
                recording_sample_rate=self.sample_rate,
                model_sample_rate=model_sample_rate,
            )


# %% Input devices
@dataclass(kw_only=True)
class InputDevice:
    """
    An audio device that can record, as sounddevice lists it.
    """

    index: int
    """
    Position of the device in sounddevice's list, which can change when devices are
    plugged in or out.
    """

    name: str
    """
    The device's name.

    Any part of it that no other device's name contains selects it.
    """

    is_default: bool
    """
    Whether the device is the system's default input.
    """

    @classmethod
    def list_available(cls) -> list[InputDevice]:
        """
        :return: Every device with at least one input channel, in sounddevice's order.
        """
        default_index = sounddevice.default.device[0]
        return [
            cls(
                index=device["index"],
                name=device["name"],
                is_default=device["index"] == default_index,
            )
            for device in sounddevice.query_devices()
            if device["max_input_channels"] > 0
        ]

    def __str__(self) -> str:
        """
        :return: The index and the name, after a star for the default input.
        """
        marker = "*" if self.is_default else " "
        return f"{marker} {self.index:>3}  {self.name}"


# %% Microphone
@dataclass(kw_only=True)
class Microphone:
    """
    An audio input device that records mono audio on request.
    """

    device: int | str | None = None
    """
    Index or name of the input device as ``python -m sounddevice`` lists it.

    ``None`` selects the system's default input.
    """

    mute_lock: Lock = field(default_factory=Lock, init=False, compare=False, repr=False)
    """
    Guards the mute counts against mutes that begin and end in several threads at once.
    """

    begun_mute_count: int = field(default=0, init=False, compare=False)
    """
    Number of :meth:`mute` statements begun since the microphone was created.
    """

    ended_mute_count: int = field(default=0, init=False, compare=False)
    """
    Number of :meth:`mute` statements ended since the microphone was created.
    """

    @contextmanager
    def mute(self) -> Iterator[None]:
        """
        Drop what the device records while the ``with`` statement runs, for example
        while the robot speaks, including every block recorded partly during it.

        Mutes may overlap, also from several threads: the microphone is muted until the
        last of them ends.
        """
        with self.mute_lock:
            self.begun_mute_count += 1
        try:
            yield
        finally:
            with self.mute_lock:
                self.ended_mute_count += 1

    def record(self, *, duration: float, sample_rate: int) -> Recording:
        """
        Record from the device, returning once the recording is complete.

        A recording that overlaps a :meth:`mute` holds only silence.

        :param duration: Length of the recording, in seconds.
        :param sample_rate: Samples per second to record, in hertz.
        :return: The recorded audio.
        :raises MicrophoneStopped: When the device stops delivering audio.
        """
        with closing(
            self.stream(sample_rate=sample_rate, block_duration=duration)
        ) as blocks:
            block = next(blocks)
        if block.size == 0:
            block = np.zeros(round(duration * sample_rate), dtype=np.float32)
        return Recording(waveform=block, sample_rate=sample_rate)

    def stream(
        self, *, sample_rate: int, block_duration: float
    ) -> Iterator[np.ndarray]:
        """
        Listen to the device until the caller stops iterating, which releases it.

        Audio that arrives while the caller is busy is kept until it is read. A block
        recorded partly or wholly during a :meth:`mute` is empty. Samples beyond
        :attr:`Recording.full_scale` are clipped to it, and lost audio is logged.

        :param sample_rate: Samples per second to record, in hertz.
        :param block_duration: Length of each block, in seconds.
        :return: Consecutive blocks of mono float32 samples, in recording order.
        :raises MicrophoneStopped: When the device stops delivering audio.
        """
        block_length = round(block_duration * sample_rate)
        arrived_blocks: Queue[np.ndarray | None] = Queue()
        mutes_ended_before_block = self.ended_mute_count

        def keep_block(
            samples: np.ndarray,
            frame_count: int,
            time: Any,
            status: sounddevice.CallbackFlags,
        ) -> None:
            """
            Queue a clipped copy of the block that just arrived, empty when a mute ran
            during any part of it, warning first when audio was lost before it.
            """
            nonlocal mutes_ended_before_block
            if status.input_overflow:
                logging.getLogger(__name__).warning(
                    f"The input device {self.device!r} lost audio before this block."
                )
            # A mute ran during the block if one began that had not ended before the
            # block started, whether it is still running, ended or began meanwhile.
            was_muted = self.begun_mute_count > mutes_ended_before_block
            mutes_ended_before_block = self.ended_mute_count
            kept_length = 0 if was_muted else frame_count
            arrived_blocks.put(
                np.clip(
                    samples[:kept_length, 0],
                    -Recording.full_scale,
                    Recording.full_scale,
                )
            )

        with sounddevice.InputStream(
            samplerate=sample_rate,
            blocksize=block_length,
            channels=1,
            dtype=np.float32,
            device=self.device,
            callback=keep_block,
            finished_callback=lambda: arrived_blocks.put(None),
        ):
            while (block := arrived_blocks.get()) is not None:
                yield block
        raise MicrophoneStopped(device=self.device)

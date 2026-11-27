"""
Recording and streaming through a stand-in audio driver, listing the input devices, and,
outside CI, the real microphone.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from cramspeech.exceptions import MicrophoneStopped
from cramspeech.microphone import InputDevice, Microphone, Recording
from cramspeech.transcription import WhisperTranscriber
from typing_extensions import Callable

from ..pytest_environment import runs_in_continuous_integration
from .dataset.non_speech import NonSpeechRecording, NonSpeechSound
from .dataset.stand_ins import (
    AudioDevicesOfADesktop,
    ReplayingInputDevice,
    SteppedInputDevice,
)


# %% Recording
def test_record_requests_the_duration_and_returns_a_mono_recording(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The driver is asked for one mono float32 channel on the chosen device, in one block
    as long as the duration at the sample rate, and that block becomes the waveform,
    tagged with its sample rate.
    """
    sample_rate = WhisperTranscriber.sample_rate
    played = NonSpeechRecording(sound=NonSpeechSound.WHITE_NOISE).recording(
        sample_rate=sample_rate
    )
    device = replay_through_the_microphone(played)
    microphone = Microphone(device="USB microphone")

    recording = microphone.record(duration=1.5, sample_rate=sample_rate)

    assert device.requests == [
        {
            "samplerate": sample_rate,
            "blocksize": 3 * sample_rate // 2,
            "channels": 1,
            "dtype": np.float32,
            "device": microphone.device,
        }
    ]
    assert np.array_equal(recording.waveform, played.waveform[: 3 * sample_rate // 2])
    assert recording.sample_rate == sample_rate
    assert not device.is_open


def test_record_is_silence_while_muted(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    A recording made while the microphone is muted holds silence of the full duration.
    """
    sample_rate = WhisperTranscriber.sample_rate
    replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    microphone = Microphone()

    with microphone.mute():
        recording = microphone.record(duration=1.0, sample_rate=sample_rate)

    assert np.array_equal(recording.waveform, np.zeros(sample_rate, dtype=np.float32))


# %% Streaming
def test_stream_opens_a_mono_stream_with_blocks_of_the_duration(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The driver is asked for one mono float32 channel on the chosen device, delivered in
    blocks as long as the block duration at the sample rate.
    """
    sample_rate = WhisperTranscriber.sample_rate
    device = replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    microphone = Microphone(device="USB microphone")

    block = next(microphone.stream(sample_rate=sample_rate, block_duration=0.25))

    assert device.requests == [
        {
            "samplerate": sample_rate,
            "blocksize": sample_rate // 4,
            "channels": 1,
            "dtype": np.float32,
            "device": microphone.device,
        }
    ]
    assert block.shape == (sample_rate // 4,)


def test_stream_keeps_every_block_in_recording_order(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    Blocks that arrive before the caller reads them, as while it transcribes, are kept
    and joined up into the audio the device delivered.
    """
    sample_rate = WhisperTranscriber.sample_rate
    recording = NonSpeechRecording(sound=NonSpeechSound.WHITE_NOISE).recording(
        sample_rate=sample_rate
    )
    replay_through_the_microphone(recording)
    block_duration = 0.25
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=block_duration)
    block_count = len(recording.waveform) // round(block_duration * sample_rate)

    streamed = np.concatenate([next(blocks) for _ in range(block_count)])

    assert np.array_equal(streamed, recording.waveform)


def test_stream_empties_blocks_that_arrive_while_muted(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    A block that arrives while the microphone is muted stays empty, also when the caller
    reads it after the microphone was unmuted.
    """
    sample_rate = WhisperTranscriber.sample_rate
    replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.WHITE_NOISE).recording(
            sample_rate=sample_rate
        )
    )
    microphone = Microphone()
    blocks = microphone.stream(sample_rate=sample_rate, block_duration=0.25)

    with microphone.mute():
        read_while_muted = next(blocks)
    read_after_unmuting = next(blocks)

    assert read_while_muted.size == 0
    assert read_after_unmuting.size == 0


def test_stream_empties_the_block_recorded_while_a_mute_ended(
    step_through_the_microphone: Callable[[Recording], SteppedInputDevice],
) -> None:
    """
    The block that was being recorded when a mute ended still holds the end of what was
    muted, such as the robot's last word, so it stays empty; the block after it is kept.
    """
    sample_rate = WhisperTranscriber.sample_rate
    block_duration = 0.25
    device = step_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    microphone = Microphone()
    blocks = microphone.stream(sample_rate=sample_rate, block_duration=block_duration)
    next(blocks)

    with microphone.mute():
        device.deliver_next_block()
        next(blocks)
    device.deliver_next_block()
    recorded_while_the_mute_ended = next(blocks)
    device.deliver_next_block()
    after_the_mute = next(blocks)

    assert recorded_while_the_mute_ended.size == 0
    assert after_the_mute.size == round(block_duration * sample_rate)


def test_stream_empties_a_block_a_whole_mute_fell_into(
    step_through_the_microphone: Callable[[Recording], SteppedInputDevice],
) -> None:
    """
    A mute shorter than a block, which began and ended while the block was recorded,
    still empties that block.
    """
    sample_rate = WhisperTranscriber.sample_rate
    device = step_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    microphone = Microphone()
    blocks = microphone.stream(sample_rate=sample_rate, block_duration=0.25)
    next(blocks)

    with microphone.mute():
        pass
    device.deliver_next_block()

    assert next(blocks).size == 0


def test_microphone_stays_muted_until_every_mute_ended(
    step_through_the_microphone: Callable[[Recording], SteppedInputDevice],
) -> None:
    """
    Mutes may overlap, for example when two threads speak: ending one of them does not
    unmute the microphone while the other still runs.
    """
    sample_rate = WhisperTranscriber.sample_rate
    device = step_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    microphone = Microphone()
    blocks = microphone.stream(sample_rate=sample_rate, block_duration=0.25)
    next(blocks)

    with microphone.mute():
        with microphone.mute():
            pass
        device.deliver_next_block()
        next(blocks)
        device.deliver_next_block()
        still_muted = next(blocks)

    assert still_muted.size == 0


def test_stream_clips_samples_to_full_scale(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    A driver that resamples or amplifies can deliver samples beyond full scale; they are
    clipped to it.
    """
    sample_rate = WhisperTranscriber.sample_rate
    block_length = sample_rate // 4
    overshooting = np.tile(
        np.array([1.5, -1.25, 0.5, -0.5], dtype=np.float32), block_length // 4
    )
    replay_through_the_microphone(
        Recording(waveform=overshooting, sample_rate=sample_rate)
    )

    block = next(Microphone().stream(sample_rate=sample_rate, block_duration=0.25))

    assert np.array_equal(
        block, np.clip(overshooting, -Recording.full_scale, Recording.full_scale)
    )


def test_stream_raises_when_the_device_stops_delivering(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    A stream that ends, such as an unplugged microphone, fails instead of waiting for
    audio forever.
    """
    sample_rate = WhisperTranscriber.sample_rate
    replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE, duration=0.5).recording(
            sample_rate=sample_rate
        )
    )
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=0.25)

    with pytest.raises(MicrophoneStopped):
        list(blocks)


def test_stream_warns_about_every_block_after_lost_audio(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Audio the device dropped is reported, and listening goes on with the next block.
    """
    sample_rate = WhisperTranscriber.sample_rate
    block_duration = 0.25
    recording = NonSpeechRecording(sound=NonSpeechSound.TONE, duration=0.5).recording(
        sample_rate=sample_rate
    )
    device = replay_through_the_microphone(recording)
    device.drops_audio = True
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=block_duration)
    block_count = len(recording.waveform) // round(block_duration * sample_rate)

    with pytest.raises(MicrophoneStopped):
        list(blocks)
    warnings = [
        record.levelno
        for record in caplog.records
        if record.name == Microphone.__module__
    ]

    assert warnings == block_count * [logging.WARNING]


def test_stream_closes_the_device_when_the_caller_stops(
    replay_through_the_microphone: Callable[[Recording], ReplayingInputDevice],
) -> None:
    """
    The device is released as soon as the caller stops iterating.
    """
    sample_rate = WhisperTranscriber.sample_rate
    device = replay_through_the_microphone(
        NonSpeechRecording(sound=NonSpeechSound.TONE).recording(sample_rate=sample_rate)
    )
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=0.25)
    next(blocks)

    blocks.close()

    assert not device.is_open


# %% Input devices
def test_only_devices_that_record_are_listed(
    audio_devices_of_a_desktop: AudioDevicesOfADesktop,
) -> None:
    """
    Every device with an input channel is listed with its index and name, and the
    system's default input is marked.
    """
    default_index = audio_devices_of_a_desktop.default_input_index
    expected = [
        InputDevice(
            index=device["index"],
            name=device["name"],
            is_default=device["index"] == default_index,
        )
        for device in audio_devices_of_a_desktop.devices
        if device["max_input_channels"] > 0
    ]

    assert InputDevice.list_available() == expected


@pytest.mark.parametrize("is_default", [False, True])
def test_input_device_reads_as_its_index_and_name(is_default: bool) -> None:
    """
    The line names the index and the name, the two ways to choose the device, after a
    star for the system's default input, as ``python -m sounddevice`` shows it.
    """
    device = InputDevice(index=6, name="USB headset", is_default=is_default)

    marker = ["*"] if is_default else []
    assert str(device).split() == [*marker, str(device.index), *device.name.split()]


# %% Real microphone
@pytest.mark.skipif(
    runs_in_continuous_integration(),
    reason="Needs a microphone.",
)
def test_record_from_the_real_microphone() -> None:
    """
    The system's default input records mono float32 samples in full scale.
    """
    sample_rate = WhisperTranscriber.sample_rate

    recording = Microphone().record(duration=0.5, sample_rate=sample_rate)

    assert recording.waveform.shape == (sample_rate // 2,)
    assert recording.waveform.dtype == np.float32
    assert np.abs(recording.waveform).max() <= 1.0
    assert recording.sample_rate == sample_rate


@pytest.mark.skipif(
    runs_in_continuous_integration(),
    reason="Needs a microphone.",
)
def test_stream_from_the_real_microphone() -> None:
    """
    The system's default input streams mono float32 blocks in full scale.
    """
    sample_rate = WhisperTranscriber.sample_rate
    blocks = Microphone().stream(sample_rate=sample_rate, block_duration=0.25)

    first_blocks = [next(blocks), next(blocks)]
    blocks.close()

    for block in first_blocks:
        assert block.shape == (sample_rate // 4,)
        assert block.dtype == np.float32
        assert np.abs(block).max() <= 1.0

"""
Fixtures that replace the microphone driver, the models, the ``piper`` program and the
loudspeaker, the recordings tests play through them, and background listeners.
"""

from __future__ import annotations

import subprocess
from threading import Thread

import pytest
import sounddevice
from cramspeech.microphone import Microphone, Recording
from cramspeech.speech_detection import SpeechDetector
from cramspeech.speech_recognizer import BackgroundListener, SpeechRecognizer
from cramspeech.transcription import WhisperTranscriber
from typing_extensions import Callable, Iterator

from .dataset.non_speech import NonSpeechRecording, NonSpeechSound, SoundSequence
from .dataset.stand_ins import (
    AudioDevicesOfADesktop,
    DefaultDevicesOfADesktop,
    LiveInputDevice,
    LoudspeakerKeepingWhatItPlays,
    ModelHearingSpeechInEverySound,
    ModelReturningSegments,
    PiperProgram,
    ReplayingInputDevice,
    SteppedInputDevice,
)


# %% Microphone driver
@pytest.fixture
def replay_through_the_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Recording], ReplayingInputDevice]:
    """
    Makes every :class:`sounddevice.InputStream` opened during a test play the given
    recording.
    """

    def replay(recording: Recording) -> ReplayingInputDevice:
        """
        :param recording: The audio the microphone delivers from now on.
        :return: The device that plays it, for inspecting its streams.
        """
        device = ReplayingInputDevice(recording=recording)
        monkeypatch.setattr(sounddevice, "InputStream", device.input_stream)
        return device

    return replay


@pytest.fixture
def stream_through_the_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Recording], LiveInputDevice]:
    """
    Makes every :class:`sounddevice.InputStream` opened during a test play the given
    recording and then silence until it is closed.
    """

    def stream(recording: Recording) -> LiveInputDevice:
        """
        :param recording: The audio the microphone delivers before the silence.
        :return: The device that plays it, for inspecting its streams.
        """
        device = LiveInputDevice(recording=recording)
        monkeypatch.setattr(sounddevice, "InputStream", device.input_stream)
        return device

    return stream


@pytest.fixture
def step_through_the_microphone(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Recording], SteppedInputDevice]:
    """
    Makes every :class:`sounddevice.InputStream` opened during a test play the given
    recording one block at a time, each when the test delivers it.
    """

    def step(recording: Recording) -> SteppedInputDevice:
        """
        :param recording: The audio the microphone delivers, block by block.
        :return: The device that plays it, for delivering its blocks.
        """
        device = SteppedInputDevice(recording=recording)
        monkeypatch.setattr(sounddevice, "InputStream", device.input_stream)
        return device

    return step


@pytest.fixture
def audio_devices_of_a_desktop(
    monkeypatch: pytest.MonkeyPatch,
) -> AudioDevicesOfADesktop:
    """
    Makes :mod:`sounddevice` report a desktop's audio devices for the duration of a
    test.
    """
    devices = AudioDevicesOfADesktop()
    monkeypatch.setattr(sounddevice, "query_devices", devices.query_devices)
    monkeypatch.setattr(
        sounddevice,
        "default",
        DefaultDevicesOfADesktop(device=(devices.default_input_index, 0)),
    )
    return devices


# %% Piper and the loudspeaker
@pytest.fixture
def piper_program(monkeypatch: pytest.MonkeyPatch) -> PiperProgram:
    """
    Replaces running the ``piper`` program with a stand-in for the duration of a test.
    """
    program = PiperProgram()
    monkeypatch.setattr(subprocess, "run", program.run)
    return program


@pytest.fixture
def loudspeaker(monkeypatch: pytest.MonkeyPatch) -> LoudspeakerKeepingWhatItPlays:
    """
    Replaces the playing calls of :mod:`sounddevice` with a loudspeaker that keeps what
    it plays, for the duration of a test.
    """
    loudspeaker = LoudspeakerKeepingWhatItPlays()
    monkeypatch.setattr(sounddevice, "play", loudspeaker.play)
    monkeypatch.setattr(sounddevice, "wait", loudspeaker.wait)
    return loudspeaker


# %% Recordings
@pytest.fixture
def one_second_of_silence() -> Recording:
    """
    One second of silence at the sample rate Whisper takes.
    """
    return NonSpeechRecording(sound=NonSpeechSound.SILENCE, duration=1.0).recording(
        sample_rate=WhisperTranscriber.sample_rate
    )


@pytest.fixture(params=list(NonSpeechSound))
def non_speech_recording(request: pytest.FixtureRequest) -> Recording:
    """
    Each :class:`NonSpeechSound` in turn, at the sample rate Whisper takes.
    """
    return NonSpeechRecording(sound=request.param).recording(
        sample_rate=WhisperTranscriber.sample_rate
    )


@pytest.fixture
def silence_then_speech_then_pause(frame_duration: float) -> SoundSequence:
    """
    Silence, a tone that stands for speech, then silence longer than a pause.
    """
    silence = NonSpeechRecording(
        sound=NonSpeechSound.SILENCE, duration=40 * frame_duration
    )
    speech = NonSpeechRecording(sound=NonSpeechSound.TONE, duration=30 * frame_duration)
    return SoundSequence(sounds=[silence, speech, silence])


# %% Whisper model
@pytest.fixture
def model_hearing_a_request() -> ModelReturningSegments:
    """
    A Whisper stand-in that hears one spoken request in every recording.
    """
    return ModelReturningSegments(texts=["Bring me a cup."])


@pytest.fixture
def transcriber_hearing() -> Callable[[ModelReturningSegments], WhisperTranscriber]:
    """
    Builds transcribers with default settings that hear what a Whisper stand-in hears.
    """

    def hearing(model: ModelReturningSegments) -> WhisperTranscriber:
        """
        :param model: Takes the place of the Whisper model.
        :return: A transcriber that runs *model*.
        """
        transcriber = WhisperTranscriber()
        transcriber.model = model
        return transcriber

    return hearing


# %% Speech detection
@pytest.fixture
def frame_duration() -> float:
    """
    Seconds of audio in one frame the speech detector scores.
    """
    return SpeechDetector.frame_length / SpeechDetector.sample_rate


@pytest.fixture
def detector_hearing_every_sound() -> SpeechDetector:
    """
    A speech detector that takes every sound for speech and silence for a pause.
    """
    detector = SpeechDetector()
    detector.model = ModelHearingSpeechInEverySound()
    return detector


# %% Speech recognizer
@pytest.fixture
def recognizer_hearing_every_sound(
    detector_hearing_every_sound: SpeechDetector,
    model_hearing_a_request: ModelReturningSegments,
    transcriber_hearing: Callable[[ModelReturningSegments], WhisperTranscriber],
) -> SpeechRecognizer:
    """
    A recognizer on the default microphone whose detector takes every sound for speech
    and whose transcriber hears :func:`model_hearing_a_request`'s text.
    """
    return SpeechRecognizer(
        microphone=Microphone(),
        transcriber=transcriber_hearing(model_hearing_a_request),
        detector=detector_hearing_every_sound,
    )


@pytest.fixture
def stop_in_time() -> Callable[[BackgroundListener], bool]:
    """
    Leaves the ``with`` statement of background listeners from another thread, so that a
    listener that never stops fails a test instead of hanging it.
    """

    def stop(listener: BackgroundListener) -> bool:
        """
        :param listener: The listening listener to stop.
        :return: Whether leaving its ``with`` statement finished within five seconds.
        """
        stopping = Thread(
            target=listener.__exit__, args=(None, None, None), daemon=True
        )
        stopping.start()
        stopping.join(timeout=5.0)
        return not stopping.is_alive()

    return stop


@pytest.fixture
def background_listener(
    recognizer_hearing_every_sound: SpeechRecognizer,
    stop_in_time: Callable[[BackgroundListener], bool],
) -> Iterator[BackgroundListener]:
    """
    A listener on :func:`recognizer_hearing_every_sound`, stopped after the test if it
    still listens, which fails when it does not stop.
    """
    listener = BackgroundListener(recognizer=recognizer_hearing_every_sound)
    yield listener
    if listener.listening is not None and not stop_in_time(listener):
        pytest.fail("The background listener did not stop.")

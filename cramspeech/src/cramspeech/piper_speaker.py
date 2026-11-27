"""
Speaking with a Piper voice on this computer, for testing without the robot.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

import numpy as np
import sounddevice
from krrood.patterns.classproperty import classproperty
from typing_extensions import Any


# %% Piper
class PiperCommandLine(StrEnum):
    """
    The parts of a ``piper`` call that speaks a sentence read from standard input.

    ``piper`` comes with the ``piper-tts`` package, for example from
    ``uv tool install piper-tts``.
    """

    PROGRAM = "piper"
    """
    The program, which must be on the ``PATH``.
    """

    MODEL = "--model"
    """
    Followed by the voice's ``.onnx`` model.
    """

    RAW_OUTPUT = "--output-raw"
    """
    Writes the speech to standard output as mono 16-bit samples, without a WAV header.
    """


@dataclass(kw_only=True)
class VoiceConfiguration:
    """
    The part of a Piper voice's ``.onnx.json`` configuration the speaker needs.
    """

    sample_rate: int
    """
    Samples per second of the voice's speech, in hertz.
    """

    @classproperty
    def file_suffix(cls) -> str:
        """
        :return: What Piper appends to a voice model's file name to name its
            configuration.
        """
        return ".json"

    @classmethod
    def from_json(cls, *, payload: dict[str, Any]) -> VoiceConfiguration:
        """
        :param payload: The parsed ``.onnx.json`` file.
        """
        return cls(sample_rate=payload["audio"]["sample_rate"])

    @classmethod
    def from_voice_model(cls, *, voice_model: Path) -> VoiceConfiguration:
        """
        :param voice_model: The voice's ``.onnx`` model.
        :return: The configuration next to the model.
        """
        path = voice_model.with_name(voice_model.name + cls.file_suffix)
        return cls.from_json(payload=json.loads(path.read_text()))


# %% Speaker
@dataclass(kw_only=True)
class Speaker:
    """
    Says sentences out loud on this computer's default output with a Piper voice.
    """

    voice_model: Path
    """
    The voice's ``.onnx`` model, with its ``.onnx.json`` configuration next to it, as
    ``python -m piper.download_voices`` saves them.
    """

    @cached_property
    def voice_configuration(self) -> VoiceConfiguration:
        """
        The configuration of :attr:`voice_model`.
        """
        return VoiceConfiguration.from_voice_model(voice_model=self.voice_model)

    def say(self, *, text: str) -> None:
        """
        Speak *text* and return once it has been played.

        :raises subprocess.CalledProcessError: When Piper fails, for example on a
            missing voice.
        """
        piper = subprocess.run(
            [
                PiperCommandLine.PROGRAM,
                PiperCommandLine.MODEL,
                str(self.voice_model),
                PiperCommandLine.RAW_OUTPUT,
            ],
            input=text.encode(),
            stdout=subprocess.PIPE,
            check=True,
        )
        speech = np.frombuffer(piper.stdout, dtype=np.int16)
        sounddevice.play(speech, samplerate=self.voice_configuration.sample_rate)
        sounddevice.wait()

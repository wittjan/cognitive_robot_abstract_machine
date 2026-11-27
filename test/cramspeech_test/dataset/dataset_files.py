"""
The data files of the cramspeech tests.
"""

from __future__ import annotations

# %% Imports
from pathlib import Path

from krrood.patterns.path_enumeration import PathEnumeration


# %% Language model outputs
class ModelOutputFile(PathEnumeration):
    """
    Text a language model writes, kept next to the tests that script a model with it.
    """

    TWO_PLANS = Path(__file__).parent / "two_plans.json"
    """
    The plans the trained model wrote for a training instruction with two tasks:
    bringing an object, and counting objects and telling the count.
    """


# %% Piper voices
class PiperVoiceFile(PathEnumeration):
    """
    A Piper voice as the speaker loads it, its model left out since tests never run it.
    """

    VOICE_MODEL = Path(__file__).parent / "voice.onnx"
    """
    The model of a voice that speaks at 22050 Hz, like the HSR's voice, whose
    configuration lies next to it.
    """

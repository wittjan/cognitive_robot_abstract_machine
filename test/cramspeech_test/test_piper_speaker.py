"""
Saying sentences on this computer with a Piper voice, with Piper and the loudspeaker
replaced by stand-ins.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from cramspeech.piper_speaker import PiperCommandLine, Speaker, VoiceConfiguration

from .dataset.dataset_files import PiperVoiceFile
from .dataset.stand_ins import LoudspeakerKeepingWhatItPlays, PiperProgram


# %% Saying a sentence
def test_say_plays_what_piper_speaks(
    piper_program: PiperProgram, loudspeaker: LoudspeakerKeepingWhatItPlays
) -> None:
    """
    The sentence goes to Piper with the voice, and the audio Piper returns is played at
    the voice's sample rate before :meth:`Speaker.say` returns.
    """
    voice_model = PiperVoiceFile.VOICE_MODEL
    speaker = Speaker(voice_model=voice_model)

    text = "I see a cup."

    speaker.say(text=text)

    [(command, sentence)] = piper_program.calls
    [(samples, sample_rate)] = loudspeaker.played
    voice = json.loads(
        Path(f"{voice_model}{VoiceConfiguration.file_suffix}").read_text()
    )
    assert command == [
        PiperCommandLine.PROGRAM,
        PiperCommandLine.MODEL,
        str(voice_model),
        PiperCommandLine.RAW_OUTPUT,
    ]
    assert sentence == text.encode()
    assert np.array_equal(samples, piper_program.samples)
    assert sample_rate == voice["audio"]["sample_rate"]
    assert loudspeaker.waited

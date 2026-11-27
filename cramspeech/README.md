# cramspeech

Speech recognition for CRAM. cramspeech listens to a microphone, finds where somebody spoke and
paused, and transcribes each utterance with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
A trained language model then turns a heard instruction into plans.

```
Microphone ──blocks──▶ SpeechDetector ──utterance──▶ WhisperTranscriber ──▶ Transcript ──▶ PlanModel
 (sounddevice)          (Silero voice activity)       (faster-whisper)                     (llama.cpp)
└──────────────────────────── SpeechRecognizer ─────────────────────────────┘
```

`SpeechRecognizer` connects the microphone, the detector and the transcriber. `BackgroundListener`
runs it in a thread while a plan goes on. Everything is plain Python and needs no ROS.

## Installation

From the repository root:

```bash
sudo apt install libportaudio2                     # the audio library sounddevice needs
uv sync --extra speech --active --inexact          # --inexact keeps what is already installed
```

The Whisper model is downloaded from Hugging Face on the first transcription and cached in
`~/.cache/huggingface/hub` (`small` is about 480 MB). To fetch it beforehand:

```bash
python -c "import faster_whisper; faster_whisper.download_model('small')"
```

## Quick start

```python
from cramspeech.microphone import InputDevice
from cramspeech.speech_recognizer import SpeechRecognizer

for device in InputDevice.list_available():        # the default input is starred
    print(device)

recognizer = SpeechRecognizer()                    # the default input and Whisper settings
print(recognizer.hear_next_utterance(timeout=10.0).text)
```

## Three ways to listen

| Call | Returns | Use it for |
|---|---|---|
| `hear(duration=5.0)` | what was said in the next 5 s | a fixed recording window |
| `hear_next_utterance(timeout=10.0)` | the first utterance, once its speaker paused | asking one question and waiting for the answer |
| `hear_utterances(stop_requested=event)` | every utterance with words in it, as it ends | listening continuously |

```python
for transcript in recognizer.hear_utterances():    # until you break out of the loop
    print(transcript.text)
    if "goodbye" in transcript.text.lower():
        break                                      # releases the microphone
```

`hear_next_utterance` waits *timeout* seconds for speech to start, and hears speech that has
started to its end.

## Listening in a demo

A CRAM demo runs its plan with `run()`, which returns once the robot is done.
`BackgroundListener` keeps listening meanwhile, in a thread of its own, and queues every sentence
until the demo reads it. In `coraplex/demos/coraplex_hsr_mujoco_demo`, next to `demo.py`:

```python
from demo import MilkTransportDemonstration

from cramspeech.speech_recognizer import BackgroundListener, SpeechRecognizer
from cramspeech.transcription import WhisperTranscriber

recognizer = SpeechRecognizer(transcriber=WhisperTranscriber(hotwords=["milk", "table"]))
print(recognizer.hear_next_utterance(timeout=30.0).text)   # "Bring the milk to the table."

listener = BackgroundListener(recognizer=recognizer)
listener.start()                                   # hears the operator while the robot works
MilkTransportDemonstration().run()
listener.stop()                                    # at the next block, also during silence
while not listener.heard.empty():
    print("Said during the transport:", listener.heard.get().text)
```

Without the thread, pass a `threading.Event` to `hear_utterances(stop_requested=event)` and set
it from another thread. Listening ends at the next block, after a transcription under way.

## Not hearing the robot itself

Wrap everything the robot says in `recognizer.pause()`. Audio that arrives meanwhile is dropped,
together with an utterance that was under way when the pause began, also when the recognizer is
still busy with an earlier utterance:

```python
with recognizer.pause():
    say_on_the_robot("I see a cup and two bottles.")
    wait_until_spoken()
```

On the HSR the sentence goes to `/talk_request` as a `tmc_voice_msgs/msg/Voice`. The robot reports
no end of speaking, so wait inside the pause until the sentence is over.

## Changing the parameters of the pipeline

Every part is a dataclass, so a parameter is a keyword argument where the part is created. A
part left out gets its defaults, so `SpeechRecognizer(detector=SpeechDetector(pause_duration=0.7))`
changes only the pause and keeps the default microphone and Whisper settings. Set
`model_size` and `compute_type` before the first transcription: Whisper is loaded once, then, with
those two. Every other setting is read each time it is used.

```python
from cramspeech.microphone import Microphone
from cramspeech.speech_detection import SpeechDetector
from cramspeech.speech_recognizer import SpeechRecognizer
from cramspeech.transcription import (
    WhisperComputeType,
    WhisperLanguage,
    WhisperModelSize,
    WhisperTranscriber,
)

recognizer = SpeechRecognizer(
    microphone=Microphone(device="ReSpeaker"),
    transcriber=WhisperTranscriber(
        model_size=WhisperModelSize.BASE,          # faster than small, less accurate
        language=WhisperLanguage.GERMAN,
        beam_size=5,
        compute_type=WhisperComputeType.FLOAT16,   # on a graphics card
        hotwords=["Toya", "dishwasher tab"],
    ),
    detector=SpeechDetector(
        speech_threshold=0.6,                      # a noisy room
        pause_duration=0.7,                        # answer sooner after the speaker stops
    ),
    block_duration=0.25,
)
```

### Microphone

| Parameter | Default | Meaning |
|---|---|---|
| `device` | `None`, the system's default input | index, or any unique part of the name `InputDevice.list_available()` shows |

### Speech detection (`SpeechDetector`)

| Parameter | Default | Meaning | Change it when |
|---|---|---|---|
| `speech_threshold` | `0.5` | Silero score from which a frame counts as speech, 0 to 1 | noise is taken for speech (raise) or quiet speakers are missed (lower) |
| `pause_duration` | `1.0` s | silence that ends an utterance | replies come too late (lower) or speakers get cut off mid-sentence (raise) |
| `kept_before_speech` | `0.4` s | audio kept before the first speech, so the first word is not clipped | first words go missing (raise) |
| `maximum_duration` | `30.0` s | longest utterance, longer speech is cut there | long instructions get cut (raise) |

### Transcription (`WhisperTranscriber`)

| Parameter | Default | Meaning | Change it when |
|---|---|---|---|
| `model_size` | `SMALL` | `TINY`, `BASE` or `SMALL` | transcription is too slow (smaller) |
| `language` | `ENGLISH` | `ENGLISH` or `GERMAN` | the speakers use the other language |
| `beam_size` | `8` | candidate transcriptions kept while decoding | transcription is too slow (lower, `1` is fastest) |
| `compute_type` | `INT8` | `INT8` (CPU), `FLOAT16` (graphics card), `FLOAT32` (most accurate, slowest) | Whisper runs on a graphics card |
| `hotwords` | none | words and phrases the speech is likely to hold | names or objects are heard as similar words |

Whisper runs on a graphics card by itself when faster-whisper finds one, and on the CPU otherwise.

### Listening (`SpeechRecognizer`)

| Parameter | Default | Meaning |
|---|---|---|
| `block_duration` | `0.25` s | audio between two searches for a finished utterance. It is also how fast a stop takes effect |

## Plans from an instruction

`PlanModel` turns a spoken instruction into plans with a trained language model in GGUF format,
run by [llama.cpp](https://github.com/ggml-org/llama.cpp):

```python
from pathlib import Path

from cramspeech.inference.plan_model import ComputeDevice, PlanModel

plan_model = PlanModel(
    model_path=Path("plan-q4.gguf"),
    device=ComputeDevice.GPU,                      # the default, falls back to the CPU
    maximum_output_token_count=1024,               # longer plans need more
)

transcript = recognizer.hear_next_utterance(timeout=10.0)
instruction_plans = plan_model.write_plans(instruction=transcript.text)
for plan in instruction_plans.plans:               # one plan per task of the instruction
    print(plan.summary)
    for step in plan.steps:
        print(" ", step.action, step.object_name, step.location, step.person, step.text)
```

- The model can only write what [`plan_grammar.gbnf`](src/cramspeech/inference/resources/plan_grammar.gbnf)
  allows: the plan format it was trained on, with any words for objects, locations and people.
- The model is loaded on the first instruction and then kept.
- llama-cpp-python, which the CRAM workspace already installs for robokudo, runs on the CPU as
  installed. With an NVIDIA graphics card and the CUDA toolkit, rebuild it once with CUDA, and
  again after every llama-cpp-python update:

  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" uv sync --extra speech --active --inexact \
      --reinstall-package llama-cpp-python --no-cache
  ```

## Testing on a laptop

`cramspeech.testing.Speaker` says sentences on the laptop with a [Piper](https://github.com/OHF-Voice/piper1-gpl)
voice, the engine the HSR speaks with, so answers and self-hearing can be tried without the
robot. It is for testing only: on the robot, speak through its speech node.

```bash
uv tool install piper-tts                          # Piper runs as its own program, `piper`
uv tool run --from piper-tts python -m piper.download_voices --data-dir voices en_GB-jenny_dioco-medium
```

```python
from pathlib import Path

from cramspeech.testing import Speaker

speaker = Speaker(voice_model=Path("voices/en_GB-jenny_dioco-medium.onnx"))
with recognizer.pause():
    speaker.say(text="Hello, I am Toya.")          # returns once the sentence was played
```

The tests replace the microphone, the models and Piper with stand-ins, so they need no audio
hardware or download. From the repository root:

```bash
CRAM_ORM_BUILD=never python -m pytest test/cramspeech_test --confcutdir=test/cramspeech_test
```

## Errors

| Exception | Raised when |
|---|---|
| `NothingHeard` | nobody starts speaking within the timeout of `hear_next_utterance` |
| `MicrophoneStopped` | the input device stops delivering audio, for example when it is unplugged |
| `SampleRateMismatch` | a recording reaches a model at a sample rate the model does not take |

All three live in `cramspeech.exceptions`.

## Quick Start

# One-time setup
python3.12 -m venv ~/silero_env
source ~/silero_env/bin/activate
pip install "numpy<2" torch

# Run
cd ~/Desktop
echo "Ваш текст здесь." > input.txt   # or place your real file
python tts_test.py
# → writes tts_output.wav (48 kHz, 16-bit mono PCM)

The Silero model downloads automatically on first run via `torch.hub`.

## Requirements

- Python 3.12
- PyTorch (CPU build is sufficient)
- NumPy **< 2** (the script checks at startup and exits clearly if violated)
- macOS Intel or any platform with working PyTorch CPU inference

## Configuration

All tunables are constants at the top of the script:

|      Constant     |        Default        |                          Role                           |
|-------------------|-----------------------|------|
|  `INPUT_FILENAME` | `input.txt`           | Source text, read from `~/Desktop`                      |
|  `OUTPUT_FILENAME`| `tts_output_grok.wav` | Output `.wav`, written to `~/Desktop`                   |
|  `TARGET_CHARS`   | `800`                 | Soft packing target per chunk                           |
|  `MAX_CHARS`      | `900`                 | Hard ceiling per chunk (Silero fails above ~1000)       |
|  `SPEAKER`        | `xenia`               | Voice name passed to Silero; never silently substituted |
|  `SAMPLE_RATE`    | `48000`               | Hz — Silero v4_ru also accepts 8000 or 24000            |
|  `DEVICE`         | `cpu`                 | Torch device; GPU not needed                            |

## How It Works

input.txt (UTF-8, BOM-tolerant)
  │
  ├─ collapse whitespace
  ├─ split into sentences (conservative regex; survives initials like "А. П.")
  ├─ pre-split any sentence > MAX_CHARS on clause punctuation, then words
  ├─ greedily pack sentences into chunks ≤ TARGET_CHARS
  │
  ├─ load Silero v4_ru model (once, on CPU)
  │
  ├─ for each chunk:
  │     synthesize → convert float32 → int16 PCM → append to .wav.part
  │
  └─ atomically rename .wav.part → .wav

#### Design choices

- **Sentence integrity over exact sizing.** A chunk slightly over target is fine; a broken sentence is not. The only exception is a single sentence exceeding `MAX_CHARS`, which gets split at the softest available boundary (`;:—,` first, then word breaks).
- **Atomic output.** Writes to a `.part` temp file. On any failure the partial file is deleted — you never find a half-finished `.wav` masquerading as complete.
- **Stream, don't accumulate.** PCM frames are flushed to disk per chunk; memory holds only one chunk's waveform at a time.
- **Fail early.** The NumPy version, input file existence, and text-emptiness checks all run before the model loads (~10 s on CPU).

## Troubleshooting

|                     Symptom                    |                             Fix                                |
|------------------------------------------------|----------------------------------------------------------------|
| `RuntimeError: NumPy 2.x detected`             | Activate the prepared venv: `source ~/silero_env/bin/activate` |
| `FileNotFoundError: Input file not found`      | Place your text at `~/Desktop/input.txt` |
| `RuntimeError: Silero rejected a …-char chunk` | A passage is too long even after fallback splitting — shorten it manually or lower 
| `Empty audio error`                            | The chunk likely contains only symbols Silero ignores           |

## License

Silero models carry their own license — see [snakers4/silero-models](https://github.com/snakers4/silero-models). This script adds no further restrictions.
```

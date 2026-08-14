#!/usr/bin/env python3.12
"""
Run:
    source ~/silero_env/bin/activate
    cd ~/Desktop
    python tts_test.py
"""

from __future__ import annotations

import re
import wave
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# User-facing configuration.
# --------------------------------------------------------------------------- #
INPUT_FILENAME = "input.txt"        # source .txt on the Desktop
OUTPUT_FILENAME = "tts_output.wav"  # result .wav on the Desktop
TARGET_CHARS = 800                  # grouping target, not a hard rule
MAX_CHARS = 900                     # hard ceiling, safely under Silero's ~1000 limit
SPEAKER = "xenia"                   # required voice; never silently substituted
SAMPLE_RATE = 48000                 # v4_ru supports 8000 / 24000 / 48000
DEVICE = torch.device("cpu")        # explicit CPU for deterministic Intel-Mac behavior


# --------------------------------------------------------------------------- #
# Pure text logic (no knowledge of Silero, WAV, or the filesystem).
# --------------------------------------------------------------------------- #
def normalize_whitespace(text: str) -> str:
    """Collapse all runs of whitespace into single spaces; punctuation untouched."""
    return re.sub(r"\s+", " ", text).strip()


_TERMINATOR = re.compile(
    r"""
    (?<=[.!?…])          # a terminal mark
    [\"'»”’)\]]*         # optional closing quotes / brackets
    \s+                  # the gap before the next sentence
    """,
    re.VERBOSE,
)

_INITIAL = re.compile(r"(?:^|\s)[A-ZА-ЯЁ]\.$")


def split_into_sentences(text: str) -> list[str]:
    """
    Split normalized text into sentences.

    Conservative by design: merging two real sentences only makes a chunk
    slightly larger, whereas splitting one real sentence would violate the
    'never mid-sentence' requirement. Treats '...', '?!', '!?' as single
    boundaries and allows trailing closing quotes/brackets after the mark.
    """
    sentences: list[str] = []
    start = 0
    for m in _TERMINATOR.finditer(text):
        candidate = text[start:m.start()].strip()
        if _INITIAL.search(text[start:m.start()]):  # skip bare initials ("А. П.")
            continue
        if candidate:
            trailing = text[m.start():m.end()].rstrip()
            sentences.append((candidate + trailing).strip())
            start = m.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def split_long_sentence(sentence: str, max_size: int = MAX_CHARS) -> list[str]:
    """
    Fallback for a single sentence longer than the TTS hard limit.

    Sentence integrity is preferred, but Silero cannot synthesize a segment
    above its symbol limit, so we split on the softest available boundaries
    (clause punctuation first, then words) to keep the cut as natural as
    possible. This branch should rarely fire on normal prose.
    """
    if len(sentence) <= max_size:
        return [sentence]

    parts = re.split(r"(?<=[;:—,])\s+", sentence)  # clause boundaries first
    pieces: list[str] = []
    for part in parts:
        if len(part) <= max_size:
            pieces.append(part)
        else:
            buf = ""  # last resort: pack whole words up to the ceiling
            for w in part.split(" "):
                if buf and len(buf) + 1 + len(w) > max_size:
                    pieces.append(buf)
                    buf = w
                else:
                    buf = f"{buf} {w}".strip()
            if buf:
                pieces.append(buf)
    return pieces


def build_chunks(sentences: list[str], target_size: int = TARGET_CHARS) -> list[str]:
    """
    Greedy sentence packing.

    - Over-long sentences are pre-split via the hard cap so no chunk can exceed
      Silero's per-call limit.
    - An empty chunk always accepts the next piece (integrity wins over target).
    - Otherwise a piece is added only if it still fits within target_size.
    - Never splits a piece across chunks; never emits an empty chunk.
    """
    safe: list[str] = []
    for s in sentences:
        safe.extend(split_long_sentence(s, MAX_CHARS))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in safe:
        added = len(sentence) + (1 if current else 0)  # +1 for joining space
        if not current:
            current, current_len = [sentence], len(sentence)
        elif current_len + added <= target_size:
            current.append(sentence)
            current_len += added
        else:
            chunks.append(" ".join(current))
            current, current_len = [sentence], len(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


# --------------------------------------------------------------------------- #
# I/O and model layers.
# --------------------------------------------------------------------------- #
def read_text(path: Path) -> str:
    """Read a UTF-8 (BOM-tolerant) text file and ensure it is non-empty."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.strip():
        raise ValueError(f"Input file is empty: {path}")
    return raw


def load_tts_model():
    """Load the Silero v4_ru model once, on CPU."""
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-models",
        model="silero_tts",
        language="ru",
        speaker="v4_ru",
    )
    model.to(DEVICE)
    return model


def synthesize_chunk(model, chunk: str) -> np.ndarray:
    """Synthesize one chunk with the required xenia voice; return a 1-D float array."""
    try:
        audio = model.apply_tts(text=chunk, speaker=SPEAKER, sample_rate=SAMPLE_RATE)
    except ValueError as e:
        raise RuntimeError(
            f"Silero rejected a {len(chunk)}-char chunk "
            f"(per-call limit / unsupported symbols).\n"
            f"First 120 chars: {chunk[:120]!r}\n"
            f"Original error: {e!r}"
        ) from e
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio).reshape(-1)  # ensure mono 1-D
    if audio.size == 0:
        raise RuntimeError(f"TTS returned empty audio for chunk: {chunk[:60]!r}...")
    return audio


def waveform_to_pcm16(waveform: np.ndarray) -> bytes:
    """Clip to [-1, 1], scale to int16, and return raw little-endian PCM bytes."""
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def check_numpy_version() -> None:
    """Friendly diagnostic: the task pins numpy < 2 for dependency compatibility."""
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError(
            f"NumPy {np.__version__} detected, but numpy < 2 is required.\n"
            "Activate the prepared environment: source ~/silero_env/bin/activate"
        )


def main() -> None:
    check_numpy_version()

    desktop = Path.home() / "Desktop"
    input_path = desktop / INPUT_FILENAME
    output_path = desktop / OUTPUT_FILENAME
    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    # 1. Prepare text and chunks BEFORE loading the heavy model.
    text = normalize_whitespace(read_text(input_path))
    sentences = split_into_sentences(text)
    chunks = build_chunks(sentences, TARGET_CHARS)
    if not chunks:
        raise ValueError("No synthesizable text found after splitting.")

    longest = max(len(c) for c in chunks)
    print(f"Input : {input_path}")
    print(f"Text  : {len(text)} chars, {len(sentences)} sentences, "
          f"{len(chunks)} chunks (longest {longest} chars)")

    # 2. Load the model once.
    print("Loading Silero (ru, xenia) on CPU ...")
    model = load_tts_model()

    # 3. Stream each chunk straight into one temporary WAV.
    try:
        with wave.open(str(temp_path), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(SAMPLE_RATE)
            for i, chunk in enumerate(chunks, start=1):
                print(f"Processing chunk {i}/{len(chunks)} ({len(chunk)} chars)")
                waveform = synthesize_chunk(model, chunk)
                wav.writeframes(waveform_to_pcm16(waveform))
    except BaseException:
        if temp_path.exists():  # never leave a partial file looking finished
            temp_path.unlink()
        raise

    # 4. Atomically promote the temp file to the final output.
    temp_path.replace(output_path)
    print(f"\nDone. Wrote {len(chunks)} chunks to: {output_path}")


if __name__ == "__main__":
    main()

import argparse
import os
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from scipy.signal import resample_poly


def load_audio(audio_path):
    """Load mono audio at 16 kHz without using the PyAV decoder."""
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )

    # Average stereo channels to mono.
    audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        audio = resample_poly(
            audio,
            16000 // divisor,
            sample_rate // divisor,
        )

    return np.ascontiguousarray(audio, dtype=np.float32)


def align_words(segments, turns):
    """Match words to speakers using the greatest timestamp overlap."""
    rows = []

    for segment in segments:
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue

            overlaps = {}

            for start, end, speaker in turns:
                overlap = max(
                    0.0,
                    min(word.end, end) - max(word.start, start),
                )
                overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

            speaker = (
                max(overlaps, key=overlaps.get)
                if overlaps
                else "UNKNOWN"
            )

            if not overlaps or overlaps[speaker] == 0:
                speaker = "UNKNOWN"

            if (
                rows
                and rows[-1]["speaker"] == speaker
                and word.start - rows[-1]["end"] <= 1.0
            ):
                rows[-1]["end"] = word.end
                rows[-1]["text"] += word.word
            else:
                rows.append({
                    "start": word.start,
                    "end": word.end,
                    "speaker": speaker,
                    "text": word.word,
                })

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with timestamps and speaker labels."
    )
    parser.add_argument("audio", type=Path, help="Path to a WAV recording")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--model", default="base", help="Whisper model size")
    parser.add_argument("--num-speakers", type=int, default=2)
    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"Audio file not found: {args.audio}")

    if args.num_speakers < 1:
        parser.error("--num-speakers must be positive.")

    token = os.getenv("HF_TOKEN")
    if not token:
        parser.error("Set the HF_TOKEN environment variable first.")

    print("Loading audio...")
    audio = load_audio(args.audio)

    print("Loading speaker diarization model...")
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )

    # Pass the loaded waveform directly to pyannote as well.
    import torch

    print("Detecting speakers...")
    output = diarizer(
        {
            "waveform": torch.from_numpy(audio).unsqueeze(0),
            "sample_rate": 16000,
        },
        num_speakers=args.num_speakers,
    )

    turns = [
        (turn.start, turn.end, speaker)
        for turn, speaker in output.speaker_diarization
    ]

    print("Loading Whisper model...")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )

    print("Transcribing audio...")
    # Passing samples directly bypasses the failing PyAV decoder.
    segments, _ = model.transcribe(
        audio,
        language=args.language,
        word_timestamps=True,
        vad_filter=True,
    )

    rows = align_words(segments, turns)

    print("\nTranscript:\n")

    if not rows:
        print("No speech was transcribed.")

    for row in rows:
        print(
            f'{row["start"]:.2f}s → {row["end"]:.2f}s '
            f'{row["speaker"]}: {row["text"].strip()}'
        )


if __name__ == "__main__":
    main()
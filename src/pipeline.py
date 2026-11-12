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
from pathlib import Path

from src.diarization import diarize_audio
from src.speaker_embedding import extract_speaker_embeddings
from src.speaker_identification import load_profile, match_speakers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diarize audio and compare its speakers to one enrolled clinician."
    )
    parser.add_argument("--audio", required=True, type=Path, help="Audio to process.")
    parser.add_argument(
        "--profile",
        required=True,
        type=Path,
        help="Clinician embedding profile created by enroll_clinician.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=2,
        help="Known speaker count; defaults to 2.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help=(
            "Experimental cosine-similarity cutoff (default: 0.75). "
            "This is not a clinically validated threshold."
        ),
    )
    args = parser.parse_args()

    if args.num_speakers < 1:
        parser.error("--num-speakers must be at least 1.")
    if not -1.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between -1 and 1.")

    profile = load_profile(args.profile)
    output = diarize_audio(args.audio, num_speakers=args.num_speakers)
    embeddings = extract_speaker_embeddings(output)
    matches = match_speakers(embeddings, profile, args.threshold)
    match_by_label = {match.speaker_label: match for match in matches}

    print(f"\nSpeaker comparison for clinician ID {profile.clinician_id!r}")
    print(f"Cosine threshold: {args.threshold:.2f} (experimental)")
    for match in matches:
        status = (
            f"CLINICIAN MATCH: {match.clinician_id}"
            if match.matched
            else "UNKNOWN / not confirmed as clinician"
        )
        print(
            f"{match.speaker_label}: {status} "
            f"(cosine similarity {match.similarity:.3f})"
        )

    print("\nDetected speaker segments:")
    for turn, speaker in output.speaker_diarization:
        match = match_by_label[speaker]
        role = "clinician" if match.matched else "unknown"
        print(f"{turn.start:.2f}s → {turn.end:.2f}s : {speaker} [{role}]")

    print(
        "\nUnmatched speakers are left unknown; they are not automatically "
        "classified as patients."
    )


if __name__ == "__main__":
    main()
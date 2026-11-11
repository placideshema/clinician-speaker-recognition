import argparse
import os

from faster_whisper import WhisperModel
from pyannote.audio import Pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", help="Path to your audio file")
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", default="base")
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    if not token:
        parser.error("Set the HF_TOKEN environment variable first.")

    if not os.path.isfile(args.audio):
        parser.error(f"Audio file not found: {args.audio}")

    print("Loading speaker diarization model...")
    diarizer = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1",
        token=token,
    )

    print("Detecting speakers...")
    output = diarizer(args.audio, num_speakers=2)
    turns = [
        (turn.start, turn.end, speaker)
        for turn, speaker in output.speaker_diarization
    ]

    print("Transcribing audio...")
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
    )
    segments, _ = model.transcribe(
        args.audio,
        language=args.language,
        word_timestamps=True,
        vad_filter=True,
    )

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
                overlaps[speaker] = overlaps.get(speaker, 0) + overlap

            speaker = max(overlaps, key=overlaps.get) if overlaps else "UNKNOWN"
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

    print("\nTranscript:\n")
    for row in rows:
        print(
            f'{row["start"]:.2f}s → {row["end"]:.2f}s '
            f'{row["speaker"]}: {row["text"].strip()}'
        )


if __name__ == "__main__":
    main()
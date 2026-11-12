import argparse
from pathlib import Path

from src.diarization import DIARIZATION_MODEL, diarize_with_pipeline, load_diarization_pipeline
from src.speaker_embedding import extract_speaker_embeddings
from src.speaker_identification import (
    create_profile,
    save_profile,
    validate_clinician_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a local speaker-embedding profile for a clinician."
    )
    parser.add_argument(
        "--audio",
        action="append",
        required=True,
        type=Path,
        help="A clean recording containing only this clinician. Repeat for more samples.",
    )
    parser.add_argument(
        "--clinician-id",
        required=True,
        help="Pseudonymous clinician ID; avoid putting a real name in the profile.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Profile output path (defaults to models/clinician_profiles/<id>.npz).",
    )
    args = parser.parse_args()

    try:
        validate_clinician_id(args.clinician_id)
    except ValueError as error:
        parser.error(str(error))

    audio_paths = [path.expanduser().resolve() for path in args.audio]
    missing_files = [path for path in audio_paths if not path.is_file()]
    if missing_files:
        parser.error("Audio file(s) not found: " + ", ".join(map(str, missing_files)))

    diarization_pipeline = load_diarization_pipeline()
    enrollment_embeddings = []
    for audio_path in audio_paths:
        output = diarize_with_pipeline(
            diarization_pipeline,
            audio_path,
            num_speakers=1,
        )
        embeddings = extract_speaker_embeddings(output)
        if len(embeddings) != 1:
            raise RuntimeError(
                f"Expected exactly one speaker in enrollment audio {audio_path}, "
                f"but found {len(embeddings)}."
            )
        enrollment_embeddings.extend(embeddings.values())

    profile = create_profile(
        args.clinician_id,
        enrollment_embeddings,
        model_id=DIARIZATION_MODEL,
    )
    project_root = Path(__file__).resolve().parents[1]
    output_path = args.output or (
        project_root
        / "models"
        / "clinician_profiles"
        / f"{args.clinician_id}.npz"
    )
    save_profile(profile, output_path)
    print(f"Saved local clinician profile to {output_path.expanduser().resolve()}")
    print(
        "The profile contains a voice embedding. Treat it as sensitive biometric "
        "data and keep it out of source control."
    )


if __name__ == "__main__":
    main()